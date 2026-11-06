from __future__ import annotations

import argparse
import ast
import concurrent.futures
import json
import logging
import traceback
from pathlib import Path
from typing import Any

import corneto as cn
from corneto.io import load_graph_from_sif_tuples
from corneto.methods import CarnivalFlow
import numpy as np
import omnipath as op
import pandas as pd

from pipeline.src.utils import build_tagged_path, configure_logging, load_yaml_file, _ensure_list

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
    return config


def build_omnipath_corneto_graph(
    config: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """
    Downloads the Prior Knowledge Network (PKN) from OmniPath, filters signed directed
    interactions, and constructs the base Corneto graph.
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
    adj_list: dict[str, list[str]] = {}

    for row in pkn.itertuples(index=False):
        src = row.source_genesymbol
        tgt = row.target_genesymbol
        sign = int(row.interaction)

        if sign != 0 and pd.notna(src) and pd.notna(tgt):
            src_str = str(src)
            tgt_str = str(tgt)
            sif_tuples.append((src_str, sign, tgt_str))
            adj_list.setdefault(src_str, []).append(tgt_str)

    # Deduplicate SIF tuples while preserving order
    sif_tuples = list(dict.fromkeys(sif_tuples))

    graph = load_graph_from_sif_tuples(sif_tuples)
    LOGGER.info(
        "Base Corneto graph created: %d vertices, %d edges.",
        graph.num_vertices,
        graph.num_edges,
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
    }
    return graph, pkn_stats


def create_corneto_data_from_csv(
    file_csv: Path,
    target_state: int = 1,
    top_tfs: int = 50,
    perturbation_node: str = "KISS1R",
) -> tuple[Any | None, pd.Series | None]:
    """
    Loads the differential TF CSV from Step 04, selects the top N TFs by absolute
    activity difference, and packages them into a Corneto Data object.
    """
    if not file_csv.exists():
        LOGGER.warning("Input CSV file '%s' not found.", file_csv)
        return None, None

    df = pd.read_csv(file_csv)
    if df.empty or "Node" not in df.columns or "Measurement" not in df.columns:
        LOGGER.warning("Input CSV file '%s' is empty or missing required columns. Skipping.", file_csv)
        return None, None

    df = df.dropna(subset=["Node", "Measurement"]).copy()
    df["Measurement"] = pd.to_numeric(df["Measurement"], errors="coerce")
    df = df.dropna(subset=["Measurement"])
    if df.empty:
        return None, None

    df["Abs_Measurement"] = df["Measurement"].abs()
    top_df = df.nlargest(top_tfs, "Abs_Measurement")

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

    return cn.Data.from_cdict({"sample_1": sample_dict}), sig_tfs


def run_carnival_simulation(
    cohort_label: str,
    corneto_data: Any | None,
    target_state: int,
    graph: Any,
    n_tfs: int,
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
    """Runs a single CARNIVAL integer/linear flow optimization for a cohort and target state."""
    LOGGER.info("[%s | %s=%d] Starting simulation...", cohort_label, perturbation_node, target_state)

    if corneto_data is None or n_tfs == 0:
        msg = f"[{cohort_label} | {perturbation_node}={target_state}] No valid TFs available. Skipped."
        LOGGER.warning(msg)
        return {
            "cohort": cohort_label,
            "target_state": target_state,
            "status": "skipped_no_tfs",
            "message": msg,
            "input_tfs_count": 0,
            "active_edges_count": 0,
            "active_nodes_count": 0,
            "edges_csv": None,
            "nodes_csv": None,
        }

    try:
        LOGGER.info(
            "[%s | %s=%d] Data ready: 1 Input (%s) -> %d Output (TFs)",
            cohort_label,
            perturbation_node,
            target_state,
            perturbation_node,
            n_tfs,
        )

        cflow = CarnivalFlow(lambda_reg=lambda_reg)
        problem = cflow.build_from_data(graph, corneto_data)

        solver_name = solver.lower()

        solver_options={
            "scipy":{"scipy_options":{"mip_rel_gap":rel_gap}},
            "gurobi":{"MIPGap":rel_gap},
            "highs":{"mip_rel_gap":rel_gap},
            "cbc":{"ratio":rel_gap}
        }

        options=solver_options.get(solver_name,{})

        problem.solve(verbosity=solver_verbosity, solver=solver, **options)

        edge_signal_vals = np.asarray(problem.expr.edge_has_signal.value).ravel()
        vertex_vals = np.asarray(problem.expr.vertex_value.value).ravel()

        active_edges_idx = np.flatnonzero(np.abs(edge_signal_vals) > active_threshold)
        active_nodes_idx = np.flatnonzero(np.abs(vertex_vals) > active_threshold)

        if len(active_edges_idx) == 0:
            msg = f"[{cohort_label} | {perturbation_node}={target_state}] No active edges found."
            LOGGER.warning(msg)
            return {
                "cohort": cohort_label,
                "target_state": target_state,
                "status": "no_active_edges",
                "message": msg,
                "input_tfs_count": n_tfs,
                "active_edges_count": 0,
                "active_nodes_count": int(len(active_nodes_idx)),
                "edges_csv": None,
                "nodes_csv": None,
            }

        all_edge_ids = list(cflow.processed_graph.edges())
        edge_interactions = cflow.processed_graph.get_attr_from_edges("interaction")
        edge_values_flat = np.asarray(problem.expr.edge_value.value).ravel()

        sources: list[str] = []
        targets: list[str] = []
        for idx in active_edges_idx:
            _, (src_set, tgt_set) = all_edge_ids[idx]
            src = str(next(iter(src_set), "Unknown"))
            tgt = str(next(iter(tgt_set), "Unknown"))
            sources.append(src)
            targets.append(tgt)

        interactions = [edge_interactions[idx] for idx in active_edges_idx]
        edge_activities = edge_values_flat[active_edges_idx]

        df_edges_active = pd.DataFrame(
            {
                "Source": sources,
                "Interaction": interactions,
                "Target": targets,
                "Activity": edge_activities,
            }
        )

        edges_csv_path = build_tagged_path(
            edges_base_name,
            cohort_label,
            perturbation_node,
            str(target_state),
            default_dir=output_dir,
        )
        df_edges_active.to_csv(edges_csv_path, index=False)

        all_vertices = list(cflow.processed_graph.vertices)
        node_names = [str(all_vertices[idx]) for idx in active_nodes_idx]
        node_activities = vertex_vals[active_nodes_idx]

        df_nodes_active = pd.DataFrame(
            {
                "Node_ID": node_names,
                "Activity": node_activities,
            }
        )

        nodes_csv_path = build_tagged_path(
            nodes_base_name,
            cohort_label,
            perturbation_node,
            str(target_state),
            default_dir=output_dir,
        )
        df_nodes_active.to_csv(nodes_csv_path, index=False)

        msg = (
            f"[{cohort_label} | {perturbation_node}={target_state}] SUCCESS - "
            f"{len(df_edges_active)} edges, {len(df_nodes_active)} nodes."
        )
        LOGGER.info(msg)

        return {
            "cohort": cohort_label,
            "target_state": target_state,
            "status": "success",
            "message": msg,
            "input_tfs_count": n_tfs,
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
            "input_tfs_count": n_tfs,
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
) -> dict[str, Path]:
    """
    Resolves cohort_label -> CSV path mapping from:
      1. Explicit CLI arguments (--input-csvs, supporting 'LABEL=path.csv' or 'path.csv'),
      2. step04_report.json if available, or
      3. Pattern scan in input_dir for TF_Measurements_CARNIVAL_*.csv.
    """
    discovered: dict[str, Path] = {}

    if explicit_csvs:
        for item in explicit_csvs:
            if "=" in item:
                label, raw_path = item.split("=", 1)
                discovered[label.strip()] = Path(raw_path.strip())
            else:
                p = Path(item.strip())
                label = p.stem.replace("TF_Measurements_CARNIVAL_", "")
                discovered[label] = p
    elif step04_report_path.exists():
        with step04_report_path.open("r", encoding="utf-8") as handle:
            step04_rep = json.load(handle)
        strat_mode = str(step04_rep.get("stratification_mode", "")).upper()
        for cohort_label, cdata in step04_rep.get("cohorts", {}).items():
            csv_str = cdata.get("decoupler_consensus", {}).get("output_csv")
            if csv_str and Path(csv_str).exists():
                key = f"{cohort_label}_{strat_mode}" if strat_mode else cohort_label
                discovered[key] = Path(csv_str)

    if not discovered and input_dir.exists():
        for csv_path in sorted(input_dir.glob("TF_Measurements_CARNIVAL_*.csv")):
            label = csv_path.stem.replace("TF_Measurements_CARNIVAL_", "")
            discovered[label] = csv_path

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
    input_files: dict[str, Path],
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
        help="Directory containing Step 04 TF measurement CSV files.",
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

    # 2. Download OmniPath PKN and build base Corneto graph
    base_graph, pkn_stats = build_omnipath_corneto_graph(config)

    # 3. Prepare tasks and execute CARNIVAL simulations in parallel
    tasks_metadata: dict[str, Any] = {}
    simulation_results: list[dict[str, Any]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: dict[concurrent.futures.Future[dict[str, Any]], str] = {}

        for cohort_label, csv_path in input_files.items():
            LOGGER.info("Queueing simulations for cohort: %s (%s)", cohort_label, csv_path)
            tasks_metadata[cohort_label] = {
                "input_csv": str(csv_path),
                "simulations": {},
            }

            for state in target_states:
                corneto_data, sig_tfs = create_corneto_data_from_csv(
                    file_csv=csv_path,
                    target_state=state,
                    top_tfs=top_tfs,
                    perturbation_node=perturbation_node,
                )
                n_tfs = int(len(sig_tfs)) if sig_tfs is not None else 0

                future = executor.submit(
                    run_carnival_simulation,
                    cohort_label=cohort_label,
                    corneto_data=corneto_data,
                    target_state=state,
                    graph=base_graph.copy(),
                    n_tfs=n_tfs,
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

        LOGGER.info("All %d simulations are running in parallel (max_workers=%d)...", len(futures), max_workers)

        for future in concurrent.futures.as_completed(futures):
            task_name = futures[future]
            try:
                res = future.result()
                simulation_results.append(res)
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