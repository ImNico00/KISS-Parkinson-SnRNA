from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import traceback
from pathlib import Path
from typing import Any

import corneto as cn
from corneto.io import load_graph_from_sif_tuples
from corneto.methods import CarnivalFlow
import networkx as nx
import numpy as np
import omnipath as op
import pandas as pd

from pipeline.src.schemas import CANONICAL_NAMES
from pipeline.src.utils import (
    _ensure_list,
    build_tagged_path,
    configure_logging,
    load_yaml_file,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_STEP05_CONFIG: dict[str, Any] = {
    "organisms": "human",
    "omnipath_resources": [
        "ARN",
        "SIGNOR",
        "SignaLink3",
        "KEGG",
        "ACSN",
        "Reactome_SignaLink3",
        "Reactome_ProtMapper",
    ],
    "omnipath_types": ["post_translational", "transcriptional"],
    "perturbation_node": "KISS1R",
    "target_states": [1, -1],
    "top_tfs": 50,
    "min_tfs_for_carnival": 5,
    # Fallback strategy if strict primary FDR yields < min_tfs_for_carnival:
    # "primary_nominal" (p < 0.05 on pre-specified primary consensus),
    # "top_continuous" (top_tfs by |primary_delta_activity| as in Liu et al. 2019),
    # or "none" (strictly use only FDR-significant TFs from TF_Measurements_*.csv)
    "fallback_tf_selection": "primary_nominal",
    "nominal_p_threshold": 0.05,
    "diff_threshold": 0.20,
    "weight_tfs_by_stability": False,
    "lambda_reg": 0.001,
    "solver": "scipy",
    "rel_gap": 0.01,
    "solver_verbosity": 0,
    "active_threshold": 1e-5,
    "max_workers": 3,
}


def _resolve_step05_config(config_path: Path) -> dict[str, Any]:
    config = DEFAULT_STEP05_CONFIG.copy()
    if config_path.exists():
        loaded = load_yaml_file(config_path) or {}
        step05_cfg = loaded.get("step05", {})
        if step05_cfg and not isinstance(step05_cfg, dict):
            raise ValueError(f"Expected mapping under 'step05' in config: {config_path}")
        config.update({k: v for k, v in step05_cfg.items() if v is not None})
    else:
        LOGGER.warning("Config file not found (%s), using default values.", config_path)

    config["omnipath_resources"] = [str(r) for r in _ensure_list(config["omnipath_resources"])]
    config["omnipath_types"] = [str(t) for t in _ensure_list(config["omnipath_types"])]
    config["target_states"] = [int(s) for s in _ensure_list(config["target_states"])]
    config["weight_tfs_by_stability"] = bool(config.get("weight_tfs_by_stability", False))
    return config


def build_omnipath_corneto_graph(
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[Any, set[str], dict[str, Any]]:
    """
    Downloads the Prior Knowledge Network (PKN) from OmniPath, filters signed directed
    interactions, constructs the base Corneto graph, and exports the PKN node universe
    and degree table for Step 06 validation.
    """
    organisms = str(config["organisms"])
    resources: list[str] = config["omnipath_resources"]
    interaction_types: list[str] = config["omnipath_types"]

    LOGGER.info(
        "Downloading OmniPath PKN (organisms=%s, resources=%s, types=%s)...",
        organisms,
        resources,
        interaction_types,
    )

    knowledge_net = op.interactions.OmniPath.get(
        organisms=organisms,
        resources=resources,
        types=interaction_types,
        genesymbols=True,
    )
    raw_interactions_count = int(len(knowledge_net))
    LOGGER.info("Extracted raw OmniPath interactions: %d", raw_interactions_count)

    pkn = knowledge_net[knowledge_net["consensus_direction"]].copy()
    pkn["interaction"] = (
        pkn["is_stimulation"].astype(int) - pkn["is_inhibition"].astype(int)
    )
    pkn = pkn[pkn["interaction"] != 0].copy()

    sif_tuples: list[tuple[str, int, str]] = []
    G_nx = nx.DiGraph()

    for row in pkn.itertuples(index=False):
        src = row.source_genesymbol
        tgt = row.target_genesymbol
        sign = int(row.interaction)

        if sign != 0 and pd.notna(src) and pd.notna(tgt):
            src_str = str(src).strip()
            tgt_str = str(tgt).strip()
            if src_str and tgt_str and src_str != tgt_str:
                sif_tuples.append((src_str, sign, tgt_str))
                G_nx.add_edge(src_str, tgt_str, sign=sign)

    # Deduplicate SIF tuples while preserving order
    sif_tuples = list(dict.fromkeys(sif_tuples))

    graph = load_graph_from_sif_tuples(sif_tuples)
    pkn_vertices = {str(v) for v in graph.vertices}

    # Export PKN universe and node degrees for Step 06 ORA and topological validation
    pkn_nodes_df = pd.DataFrame(
        {
            "Node": sorted(G_nx.nodes()),
            "in_degree": [int(G_nx.in_degree(n)) for n in sorted(G_nx.nodes())],
            "out_degree": [int(G_nx.out_degree(n)) for n in sorted(G_nx.nodes())],
            "all_degree": [int(G_nx.degree(n)) for n in sorted(G_nx.nodes())],
        }
    )
    pkn_nodes_csv = output_dir / "OmniPath_PKN_Nodes.csv"
    pkn_nodes_df.to_csv(pkn_nodes_csv, index=False)

    LOGGER.info(
        "Base Corneto graph created: %d vertices, %d edges (saved universe -> %s).",
        graph.num_vertices,
        graph.num_edges,
        pkn_nodes_csv,
    )

    pkn_stats = {
        "organisms": organisms,
        "resources": resources,
        "interaction_types": interaction_types,
        "raw_interactions": raw_interactions_count,
        "signed_directed_interactions": int(len(pkn)),
        "unique_sif_tuples": int(len(sif_tuples)),
        "graph_vertices": int(graph.num_vertices),
        "graph_edges": int(graph.num_edges),
        "pkn_nodes_csv": str(pkn_nodes_csv),
    }
    return graph, pkn_vertices, pkn_stats


def _resolve_stability_csv_for_cohort(
    cohort_label: str,
    input_dir: Path,
    measurements_csv: Path | None,
) -> Path | None:
    """Finds the matching Step 04 `TF_Stability_*.csv` file for a cohort."""
    candidates: list[Path] = []
    if measurements_csv is not None:
        stem_replaced = measurements_csv.name.replace("TF_Measurements", "TF_Stability")
        candidates.append(measurements_csv.parent / stem_replaced)

    candidates.extend(
        [
            input_dir / f"TF_Stability_{cohort_label}.csv",
            build_tagged_path(Path("TF_Stability.csv"), cohort_label, default_dir=input_dir),
        ]
    )
    return next((p for p in candidates if p.exists()), None)


def create_corneto_data_from_step04(
    cohort_label: str,
    file_csv: Path | None,
    stability_csv: Path | None,
    pkn_vertices: set[str],
    config: dict[str, Any],
    target_state: int = 1,
    top_tfs: int = 50,
    perturbation_node: str = "KISS1R",
) -> tuple[Any | None, pd.Series | None, dict[str, Any]]:
    """
    Loads Step 04 TF scores, filters TFs to those present in the OmniPath PKN,
    applies controlled fallback to `TF_Stability_*.csv` if strict FDR yielded fewer
    than `min_tfs_for_carnival`, selects the top N TFs by |Measurement|, and builds
    the Corneto Data object.
    """
    min_tfs = int(config.get("min_tfs_for_carnival", 5))
    fallback_mode = str(config.get("fallback_tf_selection", "primary_nominal")).lower()
    nom_p_thresh = float(config.get("nominal_p_threshold", 0.05))
    diff_thresh = float(config.get("diff_threshold", 0.20))
    weight_by_stability = bool(config.get("weight_tfs_by_stability", False))

    if perturbation_node not in pkn_vertices:
        LOGGER.warning(
            "[%s] Perturbation node '%s' is not present in the OmniPath PKN vertices!",
            cohort_label,
            perturbation_node,
        )

    df_selected = pd.DataFrame()
    selection_tier = "primary_fdr"

    # 1. Try loading strict primary FDR measurements CSV
    if file_csv is not None and file_csv.exists():
        df_raw = pd.read_csv(file_csv)
        if not df_raw.empty and {"Node", "Measurement"}.issubset(df_raw.columns):
            df_raw = df_raw.dropna(subset=["Node", "Measurement"]).copy()
            df_raw["Node"] = df_raw["Node"].astype(str).str.strip()
            df_raw["Measurement"] = pd.to_numeric(df_raw["Measurement"], errors="coerce")
            df_raw = df_raw.dropna(subset=["Measurement"])
            df_raw = df_raw[df_raw["Node"].isin(pkn_vertices)].copy()
            df_selected = df_raw

    # 2. Controlled fallback to Step 04 `TF_Stability_*.csv` if fewer than `min_tfs`
    if len(df_selected) < min_tfs and fallback_mode != "none" and stability_csv and stability_csv.exists():
        stab_df = pd.read_csv(stability_csv)
        if not stab_df.empty and {"Node", "primary_delta_activity"}.issubset(stab_df.columns):
            stab_df["Node"] = stab_df["Node"].astype(str).str.strip()
            stab_df = stab_df[stab_df["Node"].isin(pkn_vertices)].copy()

            if fallback_mode == "primary_nominal" and "primary_pval" in stab_df.columns:
                mask_nom = (stab_df["primary_pval"] < nom_p_thresh) & (
                    stab_df["primary_delta_activity"].abs() > diff_thresh
                )
                cand_df = stab_df.loc[mask_nom].copy()
                selection_tier = f"primary_nominal_p<{nom_p_thresh}"
                if len(cand_df) < min_tfs:
                    cand_df = stab_df[stab_df["primary_pval"] < 0.10].copy()
                    selection_tier = "primary_relaxed_p<0.10"
            elif fallback_mode == "top_continuous":
                cand_df = stab_df[stab_df["primary_delta_activity"].abs() > diff_thresh].copy()
                selection_tier = "top_continuous_nes"
            else:
                cand_df = pd.DataFrame()

            if not cand_df.empty:
                measurements = cand_df["primary_delta_activity"].astype(float)
                if weight_by_stability and "single_method_sign_concordance" in cand_df.columns:
                    concordance = cand_df["single_method_sign_concordance"].astype(float).clip(0.25, 1.0)
                    measurements = measurements * concordance
                    selection_tier += "_stability_weighted"

                df_selected = pd.DataFrame(
                    {
                        "Node": cand_df["Node"].values,
                        "Measurement": measurements.values,
                    }
                )
                LOGGER.info(
                    "[%s] Fallback TF selection activated (%s): %d PKN-mapped TFs available from %s.",
                    cohort_label,
                    selection_tier,
                    len(df_selected),
                    stability_csv.name,
                )

    if df_selected.empty:
        return None, None, {"selection_tier": "empty", "pkn_mapped_tfs": 0}

    df_selected["Abs_Measurement"] = df_selected["Measurement"].abs()
    top_df = df_selected.nlargest(top_tfs, "Abs_Measurement").copy()

    sample_dict: dict[str, dict[str, Any]] = {
        perturbation_node: {
            "value": int(target_state),
            "mapping": "vertex",
            "role": "input",
        }
    }

    for row in top_df.itertuples(index=False):
        sample_dict[str(row.Node)] = {
            "value": float(row.Measurement),
            "mapping": "vertex",
            "role": "output",
        }

    sig_tfs = pd.Series(
        data=top_df["Measurement"].to_numpy(dtype=float),
        index=top_df["Node"].astype(str).to_numpy(),
    )

    selection_meta = {
        "selection_tier": selection_tier,
        "pkn_mapped_tfs": int(len(df_selected)),
        "selected_top_tfs": int(len(sig_tfs)),
    }
    return cn.Data.from_cdict({"sample_1": sample_dict}), sig_tfs, selection_meta


def run_carnival_simulation(
    cohort_label: str,
    corneto_data: Any | None,
    sig_tfs: pd.Series | None,
    selection_meta: dict[str, Any],
    target_state: int,
    graph: Any,
    pkn_vertices: set[str],
    perturbation_node: str,
    output_dir: Path,
    edges_base_name: str,
    nodes_base_name: str,
    lambda_reg: float = 1e-3,
    solver: str = "scipy",
    rel_gap: float = 0.01,
    solver_verbosity: int = 0,
    active_threshold: float = 1e-5,
) -> dict[str, Any]:
    """
    Runs a single CARNIVAL flow optimization for a cohort and target state,
    filters out synthetic Corneto boundary edges ("Unknown"), and exports
    clean biological edges and role-annotated nodes (`Input`, `TF`, `Inferred_Mediator`).
    """
    n_tfs = int(len(sig_tfs)) if sig_tfs is not None else 0
    LOGGER.info("[%s | %s=%d] Starting simulation...", cohort_label, perturbation_node, target_state)

    if corneto_data is None or sig_tfs is None or n_tfs == 0:
        msg = f"[{cohort_label} | {perturbation_node}={target_state}] No valid TFs available. Skipped."
        LOGGER.warning(msg)
        return {
            "cohort": cohort_label,
            "target_state": target_state,
            "status": "skipped_no_tfs",
            "message": msg,
            "tf_selection": selection_meta,
            "input_tfs_count": 0,
            "explained_tfs_count": 0,
            "active_edges_count": 0,
            "active_nodes_count": 0,
            "edges_csv": None,
            "nodes_csv": None,
        }

    try:
        LOGGER.info(
            "[%s | %s=%d] Data ready: 1 Input (%s) -> %d Output TFs (tier=%s)",
            cohort_label,
            perturbation_node,
            target_state,
            perturbation_node,
            n_tfs,
            selection_meta.get("selection_tier"),
        )

        cflow = CarnivalFlow(lambda_reg=lambda_reg)
        problem = cflow.build_from_data(graph, corneto_data)

        solver_name = solver.lower()
        solver_options = {
            "scipy": {"scipy_options": {"mip_rel_gap": rel_gap}},
            "gurobi": {"MIPGap": rel_gap},
            "highs": {"mip_rel_gap": rel_gap},
            "cbc": {"ratio": rel_gap},
        }
        options = solver_options.get(solver_name, {})

        problem.solve(verbosity=solver_verbosity, solver=solver, **options)

        edge_signal_vals = np.asarray(problem.expr.edge_has_signal.value).ravel()
        vertex_vals = np.asarray(problem.expr.vertex_value.value).ravel()

        active_edges_idx = np.flatnonzero(np.abs(edge_signal_vals) > active_threshold)
        active_nodes_idx = np.flatnonzero(np.abs(vertex_vals) > active_threshold)

        all_edge_ids = list(cflow.processed_graph.edges())
        edge_interactions = cflow.processed_graph.get_attr_from_edges("interaction")
        edge_values_flat = np.asarray(problem.expr.edge_value.value).ravel()

        edge_rows: list[dict[str, Any]] = []
        for idx in active_edges_idx:
            _, (src_set, tgt_set) = all_edge_ids[idx]
            # Exclude synthetic Corneto boundary edges where source or target set is empty
            if not src_set or not tgt_set:
                continue
            src = str(next(iter(src_set))).strip()
            tgt = str(next(iter(tgt_set))).strip()
            if (
                not src
                or not tgt
                or src == "Unknown"
                or tgt == "Unknown"
                or src.startswith("_")
                or tgt.startswith("_")
            ):
                continue
            if pkn_vertices and (src not in pkn_vertices or tgt not in pkn_vertices):
                continue

            edge_rows.append(
                {
                    "Source": src,
                    "Interaction": int(edge_interactions[idx]),
                    "Target": tgt,
                    "Activity": round(float(edge_values_flat[idx]), 6),
                }
            )

        df_edges_active = pd.DataFrame(edge_rows)
        if not df_edges_active.empty:
            df_edges_active = df_edges_active.drop_duplicates(
                subset=["Source", "Interaction", "Target"]
            )

        if df_edges_active.empty:
            msg = f"[{cohort_label} | {perturbation_node}={target_state}] No biological active edges found."
            LOGGER.warning(msg)
            return {
                "cohort": cohort_label,
                "target_state": target_state,
                "status": "no_active_edges",
                "message": msg,
                "tf_selection": selection_meta,
                "input_tfs_count": n_tfs,
                "explained_tfs_count": 0,
                "active_edges_count": 0,
                "active_nodes_count": 0,
                "edges_csv": None,
                "nodes_csv": None,
            }

        edges_csv_path = build_tagged_path(
            edges_base_name,
            cohort_label,
            perturbation_node,
            str(target_state),
            default_dir=output_dir,
        )
        df_edges_active.to_csv(edges_csv_path, index=False)

        # Build clean active nodes table restricted to biological PKN nodes connected in the solution
        all_vertices = list(cflow.processed_graph.vertices)
        vertex_activity_map: dict[str, float] = {}
        for idx in active_nodes_idx:
            v_name = str(all_vertices[idx]).strip()
            if v_name and not v_name.startswith("_") and (not pkn_vertices or v_name in pkn_vertices):
                vertex_activity_map[v_name] = float(vertex_vals[idx])

        connected_nodes = set(df_edges_active["Source"]).union(set(df_edges_active["Target"]))
        tf_measurement_map = sig_tfs.to_dict()

        node_rows: list[dict[str, Any]] = []
        for node_id in sorted(connected_nodes.union(vertex_activity_map.keys())):
            act_val = vertex_activity_map.get(node_id, 0.0)
            if abs(act_val) <= active_threshold and node_id not in connected_nodes:
                continue

            if node_id == perturbation_node:
                role = "Input"
                input_meas = float(target_state)
            elif node_id in tf_measurement_map:
                role = "TF"
                input_meas = float(tf_measurement_map[node_id])
            else:
                role = "Inferred_Mediator"
                input_meas = np.nan

            node_rows.append(
                {
                    "Node_ID": node_id,
                    "Activity": round(float(act_val), 6),
                    "Role": role,
                    "Input_Measurement": None if pd.isna(input_meas) else round(input_meas, 6),
                }
            )

        df_nodes_active = pd.DataFrame(node_rows)
        nodes_csv_path = build_tagged_path(
            nodes_base_name,
            cohort_label,
            perturbation_node,
            str(target_state),
            default_dir=output_dir,
        )
        df_nodes_active.to_csv(nodes_csv_path, index=False)

        n_explained_tfs = int((df_nodes_active["Role"] == "TF").sum())
        n_inferred_mediators = int((df_nodes_active["Role"] == "Inferred_Mediator").sum())

        msg = (
            f"[{cohort_label} | {perturbation_node}={target_state}] SUCCESS - "
            f"{len(df_edges_active)} edges, {len(df_nodes_active)} nodes "
            f"({n_explained_tfs}/{n_tfs} TFs explained, {n_inferred_mediators} inferred mediators)."
        )
        LOGGER.info(msg)

        return {
            "cohort": cohort_label,
            "target_state": target_state,
            "status": "success",
            "message": msg,
            "tf_selection": selection_meta,
            "input_tfs_count": n_tfs,
            "explained_tfs_count": n_explained_tfs,
            "inferred_mediators_count": n_inferred_mediators,
            "active_edges_count": int(len(df_edges_active)),
            "active_nodes_count": int(len(df_nodes_active)),
            "edges_csv": str(edges_csv_path),
            "nodes_csv": str(nodes_csv_path),
        }

    except Exception as exc:
        err_trace = traceback.format_exc()
        msg = f"[{cohort_label} | {perturbation_node}={target_state}] ERROR: {type(exc).__name__}: {exc}"
        LOGGER.error("%s\n%s", msg, err_trace)
        return {
            "cohort": cohort_label,
            "target_state": target_state,
            "status": "error",
            "message": msg,
            "error_traceback": err_trace,
            "tf_selection": selection_meta,
            "input_tfs_count": n_tfs,
            "explained_tfs_count": 0,
            "active_edges_count": 0,
            "active_nodes_count": 0,
            "edges_csv": None,
            "nodes_csv": None,
        }


def _discover_step04_inputs(
    input_dir: Path,
    step04_report_path: Path,
    explicit_csvs: list[str] | None,
    cohorts_filter: list[str] | None,
) -> dict[str, dict[str, Path | None]]:
    """
    Resolves cohort_label -> {"measurements_csv": Path | None, "stability_csv": Path | None} from:
      1. Explicit CLI arguments (--input-csvs),
      2. step04_report.json (supporting both output_csv and stability_csv), or
      3. Pattern scan in input_dir for TF_Measurements_*.csv and TF_Stability_*.csv.
    """
    discovered: dict[str, dict[str, Path | None]] = {}

    if explicit_csvs:
        for item in explicit_csvs:
            if "=" in item:
                label, raw_path = item.split("=", 1)
                p = Path(raw_path.strip())
                label = label.strip()
            else:
                p = Path(item.strip())
                label = (
                    p.stem.replace("TF_Measurements_CARNIVAL_", "")
                    .replace("TF_Measurements_", "")
                    .replace("TF_Stability_", "")
                )
            stab_p = _resolve_stability_csv_for_cohort(label, input_dir, p)
            discovered[label] = {"measurements_csv": p, "stability_csv": stab_p}

    elif step04_report_path.exists():
        with step04_report_path.open("r", encoding="utf-8") as handle:
            step04_rep = json.load(handle)
        strat_mode = str(step04_rep.get("stratification_mode", "")).upper()
        for cohort_label, cdata in step04_rep.get("cohorts", {}).items():
            cons_block = cdata.get("decoupler_consensus", {})
            csv_str = cons_block.get("output_csv")
            stab_str = cons_block.get("stability_csv")

            meas_path = Path(csv_str) if csv_str and Path(csv_str).exists() else None
            stab_path = Path(stab_str) if stab_str and Path(stab_str).exists() else None

            if meas_path is not None or stab_path is not None:
                key = f"{cohort_label}_{strat_mode}" if strat_mode else cohort_label
                discovered[key] = {
                    "measurements_csv": meas_path,
                    "stability_csv": stab_path,
                }

    if not discovered and input_dir.exists():
        for csv_path in sorted(input_dir.glob("TF_Measurements_*.csv")):
            label = csv_path.stem.replace("TF_Measurements_CARNIVAL_", "").replace(
                "TF_Measurements_", ""
            )
            stab_p = _resolve_stability_csv_for_cohort(label, input_dir, csv_path)
            discovered[label] = {"measurements_csv": csv_path, "stability_csv": stab_p}

        if not discovered:
            for stab_path in sorted(input_dir.glob("TF_Stability_*.csv")):
                label = stab_path.stem.replace("TF_Stability_", "")
                discovered[label] = {"measurements_csv": None, "stability_csv": stab_path}

    if cohorts_filter:
        allowed = {c.upper() for c in cohorts_filter}
        discovered = {
            k: v
            for k, v in discovered.items()
            if k.upper() in allowed or any(k.upper().startswith(a) for a in allowed)
        }

    return discovered


def format_pipeline_step05_summary(
    *,
    config_path: str,
    input_files: dict[str, dict[str, Path | None]],
    perturbation_node: str,
    target_states: list[int],
    max_workers: int,
    output_dir: str,
    report_path: str,
) -> None:
    LOGGER.info(
        "Pipeline Step05 summary:\n"
        f"- Config File: {config_path}\n"
        f"- Input Cohorts ({len(input_files)}): {', '.join(input_files.keys()) if input_files else 'None'}\n"
        f"- Perturbation Node: {perturbation_node} (states={target_states})\n"
        f"- Parallel Threads (max_workers): {max_workers}\n"
        f"- Output Directory: {output_dir}\n"
        f"- Report Path: {report_path}"
    )


def build_step05_parser(default_repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step05 OmniPath PKN & Parallel CARNIVAL Network Reconstruction (Corneto)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    processed_dir = default_repo_root / "pipeline" / "data" / "processed"
    reports_dir = default_repo_root / "pipeline" / "reports"

    parser.add_argument(
        "--config",
        default=str(default_repo_root / "pipeline" / "config" / "step05.yml"),
        help="YAML config file for step05 tunables.",
    )
    parser.add_argument(
        "--input-dir",
        default=str(processed_dir / "step04_carnival"),
        help="Directory containing Step 04 TF measurement and stability CSV files.",
    )
    parser.add_argument(
        "--step04-report",
        default=str(reports_dir / "step04_report.json"),
        help="Path to step04_report.json for automatic CSV discovery.",
    )
    parser.add_argument(
        "--input-csvs",
        nargs="+",
        default=None,
        help=(
            "Explicit input CSV files from Step 04. Accepts either file paths or "
            "'LABEL=path/to/file.csv' pairs (overrides automatic discovery)."
        ),
    )
    parser.add_argument(
        "--cohorts",
        nargs="+",
        default=None,
        help="Optional list of cohort labels to filter discovered Step 04 files (e.g., PD_MALES).",
    )
    parser.add_argument(
        "--perturbation-node",
        default=None,
        help="Input perturbation gene symbol (overrides config, default: KISS1R).",
    )
    parser.add_argument(
        "--target-states",
        nargs="+",
        type=int,
        default=None,
        help="Perturbation states to simulate (overrides config, e.g., 1 -1).",
    )
    parser.add_argument(
        "--top-tfs",
        type=int,
        default=None,
        help="Maximum number of top differential TFs by |Measurement| (overrides config).",
    )
    parser.add_argument(
        "--lambda-reg",
        type=float,
        default=None,
        help="L1 regularization penalty for CarnivalFlow (overrides config).",
    )
    parser.add_argument(
        "--solver",
        default=None,
        help="Optimization solver backend for Corneto (overrides config, default: scipy).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of parallel worker threads for ThreadPoolExecutor (overrides config).",
    )
    parser.add_argument(
        "--edges-csv-name",
        default="Carnival_Edges.csv",
        help="Base filename for exported active edges CSV.",
    )
    parser.add_argument(
        "--nodes-csv-name",
        default="Carnival_Nodes.csv",
        help="Base filename for exported active nodes CSV.",
    )
    parser.add_argument("--output-dir", default=str(processed_dir / "step05_networks"))
    parser.add_argument("--reports-dir", default=str(reports_dir))
    parser.add_argument("--logging-level", default="INFO")
    return parser


def run_step05(args: argparse.Namespace) -> None:
    configure_logging(args.logging_level)
    config_path = Path(args.config)
    config = _resolve_step05_config(config_path)

    perturbation_node = (
        str(args.perturbation_node)
        if args.perturbation_node is not None
        else str(config["perturbation_node"])
    )
    target_states = (
        [int(s) for s in args.target_states]
        if args.target_states is not None
        else [int(s) for s in config["target_states"]]
    )
    top_tfs = int(args.top_tfs) if args.top_tfs is not None else int(config["top_tfs"])
    lambda_reg = (
        float(args.lambda_reg) if args.lambda_reg is not None else float(config["lambda_reg"])
    )
    solver = str(args.solver) if args.solver is not None else str(config["solver"])
    rel_gap = float(config["rel_gap"])
    solver_verbosity = int(config["solver_verbosity"])
    active_threshold = float(config["active_threshold"])
    max_workers = int(args.threads) if args.threads is not None else int(config["max_workers"])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover input CSV files from Step 04
    input_files = _discover_step04_inputs(
        input_dir=Path(args.input_dir),
        step04_report_path=Path(args.step04_report),
        explicit_csvs=args.input_csvs,
        cohorts_filter=args.cohorts,
    )

    if not input_files:
        raise FileNotFoundError(
            f"No Step 04 TF CSV files found in '{args.input_dir}' or '{args.step04_report}'. "
            "Run step04 first or pass '--input-csvs'."
        )

    # 2. Download OmniPath PKN, build base Corneto graph, and export PKN node table for Step 06
    base_graph, pkn_vertices, pkn_stats = build_omnipath_corneto_graph(config, out_dir)

    # 3. Prepare tasks and execute CARNIVAL simulations in parallel
    tasks_metadata: dict[str, Any] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: dict[concurrent.futures.Future[dict[str, Any]], str] = {}

        for cohort_label, paths_dict in input_files.items():
            meas_csv = paths_dict.get("measurements_csv")
            stab_csv = paths_dict.get("stability_csv")
            LOGGER.info(
                "Queueing simulations for cohort: %s (measurements=%s, stability=%s)",
                cohort_label,
                meas_csv,
                stab_csv,
            )
            tasks_metadata[cohort_label] = {
                "input_csv": str(meas_csv) if meas_csv else None,
                "stability_csv": str(stab_csv) if stab_csv else None,
                "simulations": {},
            }

            for state in target_states:
                corneto_data, sig_tfs, selection_meta = create_corneto_data_from_step04(
                    cohort_label=cohort_label,
                    file_csv=meas_csv,
                    stability_csv=stab_csv,
                    pkn_vertices=pkn_vertices,
                    config=config,
                    target_state=state,
                    top_tfs=top_tfs,
                    perturbation_node=perturbation_node,
                )

                future = executor.submit(
                    run_carnival_simulation,
                    cohort_label=cohort_label,
                    corneto_data=corneto_data,
                    sig_tfs=sig_tfs,
                    selection_meta=selection_meta,
                    target_state=state,
                    graph=base_graph.copy(),
                    pkn_vertices=pkn_vertices,
                    perturbation_node=perturbation_node,
                    output_dir=out_dir,
                    edges_base_name=args.edges_csv_name,
                    nodes_base_name=args.nodes_csv_name,
                    lambda_reg=lambda_reg,
                    solver=solver,
                    rel_gap=rel_gap,
                    solver_verbosity=solver_verbosity,
                    active_threshold=active_threshold,
                )

                task_name = f"{cohort_label} ({perturbation_node} state: {state})"
                futures[future] = task_name

        LOGGER.info(
            "All %d simulations are running in parallel (max_workers=%d)...",
            len(futures),
            max_workers,
        )

        for future in concurrent.futures.as_completed(futures):
            task_name = futures[future]
            try:
                res = future.result()
                cohort_key = res["cohort"]
                state_key = f"state_{res['target_state']}"
                tasks_metadata[cohort_key]["simulations"][state_key] = res
                LOGGER.info("Completed task: %s -> %s", task_name, res["status"])
            except Exception as exc:
                LOGGER.error("Unhandled exception in task '%s': %s", task_name, exc)

    report = {
        "config_path": str(config_path),
        "parameters": {
            "perturbation_node": perturbation_node,
            "target_states": target_states,
            "top_tfs": top_tfs,
            "min_tfs_for_carnival": int(config["min_tfs_for_carnival"]),
            "fallback_tf_selection": str(config["fallback_tf_selection"]),
            "weight_tfs_by_stability": bool(config["weight_tfs_by_stability"]),
            "lambda_reg": lambda_reg,
            "solver": solver,
            "rel_gap": rel_gap,
            "active_threshold": active_threshold,
            "max_workers": max_workers,
        },
        "omnipath_pkn": pkn_stats,
        "cohorts": tasks_metadata,
    }

    report_path = reports_dir / "step05_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    format_pipeline_step05_summary(
        config_path=str(config_path),
        input_files=input_files,
        perturbation_node=perturbation_node,
        target_states=target_states,
        max_workers=max_workers,
        output_dir=str(out_dir),
        report_path=str(report_path),
    )