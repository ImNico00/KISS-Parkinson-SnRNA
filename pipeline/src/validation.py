from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import anndata as ad
import decoupler as dc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import seaborn as sns
from scipy import stats

from pipeline.src.schemas import CANONICAL_NAMES
from pipeline.src.utils import _ensure_list, configure_logging, load_yaml_file

LOGGER = logging.getLogger(__name__)

DEFAULT_STEP06_CONFIG: dict[str, Any] = {
    "dpi_figures": 300,
    "cohorts": ["pd_male"],
    "min_cells_pseudobulk": 30,
    "target_sum_norm": 10000.0,
    "kiss1r_gene": "KISS1R",
    "max_zero_fraction_for_median": 0.50,
    "min_subjects_per_group": 4,
    "run_detected_only_subsplit": True,
    "progeny_top_genes": 500,
    "min_pathway_carnival_overlap": 1,
    "exclude_kiss1r_from_pathway_overlap": True,
    "exclude_kiss1r_from_subnetwork_score": True,
    "inherit_step04_consensus": True,
    "fallback_consensus_methods": ["aucell", "gsva", "zscore"],
    "tmin_pathway_targets": 5,
    "fdr_threshold": 0.05,
}

def _find_companion_carnival_file(
    given_path: Path,
    looking_for: str,  # "edges" or "nodes"
) -> Path | None:
    """
    Given `Carnival_Nodes_<suffix>.csv`, finds the companion `Carnival_Network_<suffix>.csv`
    or `Carnival_Edges_<suffix>.csv` in the same directory (and vice versa).
    """
    parent = given_path.parent
    name = given_path.name

    if looking_for == "edges":
        for prefix in ("Carnival_Network_", "Carnival_Edges_", "carnival_network_", "carnival_edges_"):
            if "Nodes_" in name:
                cand = parent / name.replace("Carnival_Nodes_", prefix).replace("carnival_nodes_", prefix)
                if cand.exists():
                    return cand
        # Fallback: inspect other CSVs in the same directory with the same scenario suffix (+1 / -1)
        is_neg = any(tok in name.lower() for tok in ("_-1", "neg1", "minus1"))
        for csv_p in sorted(parent.glob("*.csv")):
            if csv_p == given_path:
                continue
            c_low = csv_p.name.lower()
            c_neg = any(tok in c_low for tok in ("_-1", "neg1", "minus1"))
            if c_neg == is_neg:
                try:
                    cols = set(pd.read_csv(csv_p, nrows=1).columns)
                    if {"Source", "Interaction", "Target"}.issubset(cols):
                        return csv_p
                except Exception:
                    pass
    elif looking_for == "nodes":
        for prefix in ("Carnival_Network_", "Carnival_Edges_", "carnival_network_", "carnival_edges_"):
            if prefix in name:
                cand = parent / name.replace(prefix, "Carnival_Nodes_")
                if cand.exists():
                    return cand
    return None


def _parse_carnival_network_csv(
    carnival_csv_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads CARNIVAL nodes and edges whether `carnival_csv_path` points to:
      - A network/edges CSV (`Source, Interaction, Target, Activity`), OR
      - A nodes CSV (`Node_ID, Activity` or `Node, Activity`), automatically pairing it
        with its companion network/edges CSV in the same folder.
    """
    if not carnival_csv_path.exists():
        raise FileNotFoundError(f"CARNIVAL network CSV not found: {carnival_csv_path}")

    df = pd.read_csv(carnival_csv_path)
    cols = set(df.columns)

    nodes_df_raw: pd.DataFrame | None = None
    edges_df_raw: pd.DataFrame | None = None

    if {"Source", "Interaction", "Target"}.issubset(cols):
        edges_df_raw = df
        comp_nodes = _find_companion_carnival_file(carnival_csv_path, looking_for="nodes")
        if comp_nodes and comp_nodes.exists():
            nodes_df_raw = pd.read_csv(comp_nodes)
    elif ("Node_ID" in cols or "Node" in cols) and "Activity" in cols:
        nodes_df_raw = df
        comp_edges = _find_companion_carnival_file(carnival_csv_path, looking_for="edges")
        if comp_edges and comp_edges.exists():
            LOGGER.info("Loaded companion CARNIVAL edges file: %s", comp_edges)
            edges_df_raw = pd.read_csv(comp_edges)
    else:
        raise ValueError(
            f"Unexpected columns in {carnival_csv_path}: {list(df.columns)} "
            "(expected either ['Source', 'Interaction', 'Target', 'Activity'] or ['Node_ID', 'Activity'])"
        )

    nodes_map: dict[str, float] = {}

    # 1. Populate nodes from nodes_df_raw if available
    if nodes_df_raw is not None:
        id_col = "Node_ID" if "Node_ID" in nodes_df_raw.columns else "Node"
        for _, row in nodes_df_raw.iterrows():
            node_name = str(row[id_col]).strip()
            act = float(row["Activity"]) if pd.notna(row["Activity"]) else np.nan
            if node_name and node_name.lower() != "unknown" and not np.isnan(act) and abs(act) > 1e-9:
                nodes_map[node_name] = float(np.round(act, 6))

    # 2. Populate/augment nodes and extract edges from edges_df_raw if available
    if edges_df_raw is not None:
        if "Activity" in edges_df_raw.columns:
            for _, row in edges_df_raw.iterrows():
                tgt = str(row["Target"]).strip()
                act = float(row["Activity"]) if pd.notna(row["Activity"]) else np.nan
                if tgt and tgt.lower() != "unknown" and not np.isnan(act) and abs(act) > 1e-9:
                    nodes_map.setdefault(tgt, float(np.round(act, 6)))

        edges_mask = (
            edges_df_raw["Source"].notna()
            & (edges_df_raw["Source"].astype(str).str.strip().str.lower() != "unknown")
            & edges_df_raw["Target"].notna()
            & (edges_df_raw["Target"].astype(str).str.strip().str.lower() != "unknown")
        )
        edges_df = edges_df_raw.loc[edges_mask].copy()
        edges_df["Source"] = edges_df["Source"].astype(str).str.strip()
        edges_df["Target"] = edges_df["Target"].astype(str).str.strip()
        edges_df["Interaction"] = pd.to_numeric(edges_df["Interaction"], errors="coerce").round(6)
        if "Activity" in edges_df.columns:
            edges_df["Target_Activity"] = pd.to_numeric(edges_df["Activity"], errors="coerce").round(6)
        else:
            edges_df["Target_Activity"] = edges_df["Target"].map(nodes_map).astype(float)
        edges_df["Source_Activity"] = edges_df["Source"].map(nodes_map).astype(float)
    else:
        edges_df = pd.DataFrame(
            columns=["Source", "Interaction", "Target", "Activity", "Target_Activity", "Source_Activity"]
        )

    nodes_df = pd.DataFrame(
        [
            {"Node": gene, "Activity": act, "Sign": float(np.sign(act))}
            for gene, act in sorted(nodes_map.items())
        ]
    )
    return nodes_df, edges_df

def _resolve_carnival_csv_paths(
    pos1_arg: str | Path | None,
    neg1_arg: str | Path | None,
    step05_report_path: Path | None,
    processed_dir: Path,
    cohort_name: str = "pd_male",
) -> tuple[Path, Path | None]:
    pos1_cand = Path(pos1_arg) if pos1_arg else None
    neg1_cand = Path(neg1_arg) if neg1_arg else None

    if pos1_cand and pos1_cand.exists():
        resolved_neg1 = neg1_cand if (neg1_cand and neg1_cand.exists()) else None
        return pos1_cand, resolved_neg1

    search_dirs = [
        processed_dir / "step05_networks",
        processed_dir / "step05_carnival",
        processed_dir,
    ]
    if pos1_cand:
        search_dirs.insert(0, pos1_cand.parent)

    available_csvs: list[Path] = []
    for d in search_dirs:
        if d.exists() and d.is_dir():
            for csv_file in sorted(d.glob("*.csv")):
                if csv_file not in available_csvs:
                    available_csvs.append(csv_file)

    # Prefer network/edges files first, then nodes files (since _parse_carnival_network_csv pairs both)
    available_csvs.sort(key=lambda p: (0 if "network" in p.name.lower() or "edges" in p.name.lower() else 1, p.name))

    found_pos1: Path | None = None
    found_neg1: Path | None = None

    for csv_file in available_csvs:
        name_low = csv_file.name.lower()
        if "tf_measurements" in name_low or "step06" in name_low:
            continue
        is_neg = any(tok in name_low for tok in ("_-1", "neg1", "minus1", "kiss1r_-1"))
        is_pos = (not is_neg) and any(tok in name_low for tok in ("_1.", "_pos1", "plus1", "+1", "kiss1r_1"))

        if is_pos and found_pos1 is None:
            found_pos1 = csv_file
        elif is_neg and found_neg1 is None:
            found_neg1 = csv_file

    if found_pos1 is None:
        avail_str = ", ".join(str(p) for p in available_csvs) if available_csvs else "none found"
        raise FileNotFoundError(
            f"CARNIVAL +1 network CSV not found. Available CSVs: [{avail_str}]."
        )

    LOGGER.info("Auto-resolved CARNIVAL CSVs -> POS1: %s | NEG1: %s", found_pos1, found_neg1)
    return found_pos1, found_neg1

def _resolve_step06_config(
    config_path: Path,
    step04_config_path: Path | None = None,
) -> dict[str, Any]:
    """Loads step06.yml and synchronizes shared pseudobulk parameters with step04.yml if available."""
    config = DEFAULT_STEP06_CONFIG.copy()

    if step04_config_path and step04_config_path.exists():
        loaded_s04 = load_yaml_file(step04_config_path) or {}
        s04_cfg = loaded_s04.get("step04", {})
        if isinstance(s04_cfg, dict):
            for shared_key in ("min_cells_pseudobulk", "target_sum_norm", "kiss1r_gene"):
                if s04_cfg.get(shared_key) is not None:
                    config[shared_key] = s04_cfg[shared_key]

    if config_path.exists():
        loaded = load_yaml_file(config_path) or {}
        step06_cfg = loaded.get("step06", {})
        if step06_cfg and not isinstance(step06_cfg, dict):
            raise ValueError(f"Expected mapping under 'step06' in config: {config_path}")
        config.update({k: v for k, v in step06_cfg.items() if v is not None})
    else:
        LOGGER.warning("Step06 config file not found (%s), using defaults.", config_path)

    config["cohorts"] = [str(c) for c in _ensure_list(config["cohorts"])]
    config["fallback_consensus_methods"] = [
        str(m) for m in _ensure_list(config["fallback_consensus_methods"])
    ]
    return config


def _extract_1d_expr(adata_obj: ad.AnnData, gene: str) -> np.ndarray:
    """Extracts a 1D float numpy array of expression values for `gene` across `adata_obj.obs_names`."""
    if gene not in adata_obj.var_names:
        return np.zeros(adata_obj.n_obs, dtype=float)
    x_slice = adata_obj[:, gene].X
    if sp.issparse(x_slice):
        return np.asarray(x_slice.toarray(), dtype=float).ravel()
    return np.asarray(x_slice, dtype=float).ravel()


def _benjamini_hochberg_fdr(pvals: np.ndarray | pd.Series) -> np.ndarray:
    """Applies Benjamini-Hochberg FDR correction ignoring NaNs."""
    arr = np.asarray(pvals, dtype=float)
    n = len(arr)
    out = np.full(n, np.nan, dtype=float)
    valid = ~np.isnan(arr)
    if not np.any(valid):
        return out
    pv = arr[valid]
    m = len(pv)
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    ranked = np.clip(ranked, 0.0, 1.0)
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(m)
    out[valid] = ranked[inv_order]
    return out


def _filter_cohort_cells(adata: ad.AnnData, cohort_name: str) -> ad.AnnData:
    """Filters single-nucleus AnnData to the requested cohort (e.g., 'pd_male' / 'PD_MALES')."""
    key = cohort_name.strip().lower()
    cond_s = adata.obs.get("condition_id", pd.Series("", index=adata.obs_names)).astype(str)
    sex_s = adata.obs.get("sex", pd.Series("", index=adata.obs_names)).astype(str)

    if key in {"pd_male", "pd_males"}:
        mask = (cond_s.str.upper() == "PD") & (sex_s.str.lower() == "male")
    elif key in {"pd_female", "pd_females"}:
        mask = (cond_s.str.upper() == "PD") & (sex_s.str.lower() == "female")
    elif key in {"pd_all", "pd"}:
        mask = cond_s.str.upper() == "PD"
    elif key in {"control_male", "ctrl_male", "control_males"}:
        mask = (cond_s.str.lower() == "control") & (sex_s.str.lower() == "male")
    else:
        raise ValueError(f"Unsupported cohort key: {cohort_name}")

    return adata[mask.to_numpy(dtype=bool)].copy()


def prepare_subject_pseudobulk(
    adata_cohort: ad.AnnData,
    config: dict[str, Any],
) -> tuple[ad.AnnData, ad.AnnData, pd.DataFrame, dict[str, Any]]:
    """
    Aggregates single-nucleus profiles into subject-level (`ASAP_subject_id`) pseudobulk
    profiles using the exact Step04 cell-count threshold and normalization, analyzes the
    distribution and zero-inflation of endogenous KISS1R expression, and assigns subjects
    to `KISS1R-High` vs `KISS1R-Low` without relying on surrogate proxy genes.
    """
    min_cells = int(config["min_cells_pseudobulk"])
    target_sum = float(config["target_sum_norm"])
    kiss1r_gene = str(config["kiss1r_gene"])
    max_zero_frac = float(config.get("max_zero_fraction_for_median", 0.50))
    min_per_group = int(config.get("min_subjects_per_group", 4))

    if "ASAP_subject_id" not in adata_cohort.obs.columns:
        raise KeyError("Required column 'ASAP_subject_id' is missing from adata_cohort.obs.")

    subj_series = adata_cohort.obs["ASAP_subject_id"].astype(str)
    cell_counts = subj_series.value_counts()
    valid_subjects = sorted(cell_counts[cell_counts >= min_cells].index.tolist())

    if len(valid_subjects) < 2:
        raise ValueError(
            f"Only {len(valid_subjects)} subjects have >= {min_cells} cells; "
            "cannot perform subject-level pseudobulk validation."
        )

    adata_sub = adata_cohort[subj_series.isin(valid_subjects).to_numpy(dtype=bool)].copy()
    kiss1r_cell_expr = _extract_1d_expr(adata_sub, kiss1r_gene)

    # Sum raw counts per subject (statistical unit = ASAP_subject_id)
    n_subj = len(valid_subjects)
    n_genes = adata_sub.n_vars
    raw_mat = np.zeros((n_subj, n_genes), dtype=np.float64)
    obs_rows: list[dict[str, Any]] = []

    covar_cols = [
        c
        for c in ["age", "duration_pmi", "hoehn_yahr_stage", "path_braak_asyn", "condition_id", "sex"]
        if c in adata_sub.obs.columns
    ]

    for idx, subj_id in enumerate(valid_subjects):
        s_mask = (adata_sub.obs["ASAP_subject_id"].astype(str) == subj_id).to_numpy(dtype=bool)
        x_sub = adata_sub.X[s_mask]
        summed = np.asarray(x_sub.sum(axis=0), dtype=np.float64).ravel()
        raw_mat[idx, :] = summed

        k_cells = kiss1r_cell_expr[s_mask]
        n_pos_cells = int(np.sum(k_cells > 0))
        n_tot_cells = int(np.sum(s_mask))

        row_meta: dict[str, Any] = {
            "ASAP_subject_id": subj_id,
            "n_cells": n_tot_cells,
            "total_counts": float(np.sum(summed)),
            "kiss1r_positive_cells": n_pos_cells,
            "kiss1r_positive_cell_fraction": float(n_pos_cells / n_tot_cells) if n_tot_cells > 0 else 0.0,
        }
        sub_obs_first = adata_sub.obs.loc[s_mask].iloc[0]
        for c in covar_cols:
            row_meta[c] = sub_obs_first[c]
        obs_rows.append(row_meta)

    obs_pb = pd.DataFrame(obs_rows).set_index("ASAP_subject_id")
    obs_pb.index.name = "ASAP_subject_id"
    var_pb = adata_sub.var.copy()

    pdata_raw = ad.AnnData(X=raw_mat.astype(np.float32), obs=obs_pb.copy(), var=var_pb.copy())
    pdata_norm = pdata_raw.copy()
    sc.pp.normalize_total(pdata_norm, target_sum=target_sum)
    sc.pp.log1p(pdata_norm)

    # Extract raw and normalized KISS1R pseudobulk expression per subject
    kiss1r_raw = _extract_1d_expr(pdata_raw, kiss1r_gene)
    kiss1r_norm = _extract_1d_expr(pdata_norm, kiss1r_gene)
    detected_mask = kiss1r_norm > 0.0

    n_total = len(kiss1r_norm)
    n_detected = int(np.sum(detected_mask))
    n_undetected = int(n_total - n_detected)
    zero_fraction = float(n_undetected / n_total) if n_total > 0 else 1.0
    global_median = float(np.median(kiss1r_norm))
    pos_vals = kiss1r_norm[detected_mask]
    detected_median = float(np.median(pos_vals)) if len(pos_vals) > 0 else 0.0

    # Adaptive Statistical Stratification Rule (Phase 1)
    if zero_fraction < max_zero_frac and global_median > 0.0:
        high_mask = kiss1r_norm >= global_median
        strategy_used = "global_median_split"
        threshold_used = global_median
        rationale = (
            f"Zero fraction ({zero_fraction:.1%}) is below {max_zero_frac:.0%} and global median "
            f"({global_median:.4f}) is strictly positive; applied global median split."
        )
    else:
        high_mask = detected_mask.copy()
        strategy_used = "zero_inflated_detected_vs_undetected"
        threshold_used = 0.0
        rationale = (
            f"Distribution is zero-inflated ({n_undetected}/{n_total} subjects = {zero_fraction:.1%} "
            f"have KISS1R = 0; global median = {global_median:.4f}). A blind median split would be "
            f"degenerate; primary comparison classifies KISS1R-detected (> 0, n={n_detected}) as "
            f"KISS1R-High and KISS1R-undetected (= 0, n={n_undetected}) as KISS1R-Low, complemented "
            f"by continuous Spearman correlation across all {n_total} subjects."
        )

    n_high = int(np.sum(high_mask))
    n_low = int(n_total - n_high)
    primary_comparison_valid = (n_high >= min_per_group) and (n_low >= min_per_group)

    # Secondary sub-split restricted to KISS1R-detected subjects (if requested)
    detected_subsplit = np.full(n_total, "Undetected", dtype=object)
    if len(pos_vals) >= 2:
        for i in range(n_total):
            if detected_mask[i]:
                detected_subsplit[i] = (
                    "Detected-High" if kiss1r_norm[i] >= detected_median else "Detected-Low"
                )
    n_det_high = int(np.sum(detected_subsplit == "Detected-High"))
    n_det_low = int(np.sum(detected_subsplit == "Detected-Low"))
    detected_subsplit_valid = (n_det_high >= min_per_group) and (n_det_low >= min_per_group)

    subject_df = obs_pb.copy()
    subject_df["kiss1r_raw_counts"] = kiss1r_raw
    subject_df["kiss1r_log1p_cpm"] = kiss1r_norm
    subject_df["kiss1r_detected"] = detected_mask
    subject_df["primary_group"] = np.where(high_mask, "KISS1R-High", "KISS1R-Low")
    subject_df["classification_strategy"] = strategy_used
    subject_df["classification_threshold"] = float(threshold_used)
    subject_df["detected_subsplit_group"] = detected_subsplit
    subject_df["detected_subsplit_valid"] = bool(detected_subsplit_valid)

    # Synchronize into pdata_raw.obs and pdata_norm.obs
    for col in [
        "kiss1r_raw_counts",
        "kiss1r_log1p_cpm",
        "kiss1r_detected",
        "primary_group",
        "classification_strategy",
        "classification_threshold",
        "detected_subsplit_group",
    ]:
        pdata_raw.obs[col] = subject_df[col]
        pdata_norm.obs[col] = subject_df[col]

    strat_summary = {
        "kiss1r_gene": kiss1r_gene,
        "n_subjects_total": n_total,
        "n_kiss1r_detected": n_detected,
        "n_kiss1r_undetected": n_undetected,
        "zero_fraction": round(zero_fraction, 4),
        "global_mean_log1p_cpm": round(float(np.mean(kiss1r_norm)), 4),
        "global_sd_log1p_cpm": round(float(np.std(kiss1r_norm, ddof=1)), 4) if n_total > 1 else 0.0,
        "global_median_log1p_cpm": round(global_median, 4),
        "global_min_log1p_cpm": round(float(np.min(kiss1r_norm)), 4),
        "global_max_log1p_cpm": round(float(np.max(kiss1r_norm)), 4),
        "detected_only_mean_log1p_cpm": round(float(np.mean(pos_vals)), 4) if len(pos_vals) > 0 else 0.0,
        "detected_only_sd_log1p_cpm": (
            round(float(np.std(pos_vals, ddof=1)), 4) if len(pos_vals) > 1 else 0.0
        ),
        "detected_only_median_log1p_cpm": round(detected_median, 4),
        "classification_strategy": strategy_used,
        "classification_threshold": round(float(threshold_used), 4),
        "classification_rationale": rationale,
        "n_kiss1r_high": n_high,
        "n_kiss1r_low": n_low,
        "primary_comparison_statistically_valid": primary_comparison_valid,
        "detected_only_subsplit": {
            "n_detected_high": n_det_high,
            "n_detected_low": n_det_low,
            "threshold_detected_median": round(detected_median, 4),
            "statistically_valid": detected_subsplit_valid,
            "note": (
                "Adequate sample size for secondary sub-split."
                if detected_subsplit_valid
                else (
                    f"Underpowered ({n_det_high} vs {n_det_low} < min_subjects_per_group={min_per_group}); "
                    "annotated in CSV but inferential test not forced."
                )
            ),
        },
    }
    LOGGER.info(
        "KISS1R stratification (%s): High=%d | Low=%d (zero_fraction=%.1f%%, valid=%s)",
        strategy_used,
        n_high,
        n_low,
        zero_fraction * 100.0,
        primary_comparison_valid,
    )
    return pdata_raw, pdata_norm, subject_df.reset_index(), strat_summary


def _parse_carnival_network_csv(
    carnival_csv_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads a CARNIVAL output CSV (`Source,Interaction,Target,Activity`) and extracts:
      - `nodes_df`: unique biological nodes (`Node`, `Activity`, `Sign`), excluding `Unknown`.
      - `edges_df`: directed causal edges (`Source`, `Interaction`, `Target`, `Target_Activity`),
        excluding rows where `Source == 'Unknown'`.
    """
    if not carnival_csv_path.exists():
        raise FileNotFoundError(f"CARNIVAL network CSV not found: {carnival_csv_path}")

    df = pd.read_csv(carnival_csv_path)
    required = {"Source", "Interaction", "Target", "Activity"}
    if not required.issubset(set(df.columns)):
        raise ValueError(
            f"Unexpected columns in {carnival_csv_path}: {list(df.columns)} (expected {sorted(required)})"
        )

    nodes_map: dict[str, float] = {}
    for _, row in df.iterrows():
        tgt = str(row["Target"]).strip()
        act = float(row["Activity"]) if pd.notna(row["Activity"]) else np.nan
        if tgt and tgt.lower() != "unknown" and not np.isnan(act) and abs(act) > 1e-9:
            nodes_map[tgt] = float(np.round(act, 6))

    nodes_df = pd.DataFrame(
        [
            {"Node": gene, "Activity": act, "Sign": float(np.sign(act))}
            for gene, act in sorted(nodes_map.items())
        ]
    )

    edges_mask = (
        df["Source"].notna()
        & (df["Source"].astype(str).str.strip().str.lower() != "unknown")
        & df["Target"].notna()
        & (df["Target"].astype(str).str.strip().str.lower() != "unknown")
    )
    edges_df = df.loc[edges_mask, ["Source", "Interaction", "Target", "Activity"]].copy()
    edges_df["Source"] = edges_df["Source"].astype(str).str.strip()
    edges_df["Target"] = edges_df["Target"].astype(str).str.strip()
    edges_df["Interaction"] = pd.to_numeric(edges_df["Interaction"], errors="coerce").round(6)
    edges_df["Target_Activity"] = pd.to_numeric(edges_df["Activity"], errors="coerce").round(6)
    edges_df["Source_Activity"] = edges_df["Source"].map(nodes_map).astype(float)
    return nodes_df, edges_df


def build_pathway_network_for_carnival(
    nodes_pos1_df: pd.DataFrame,
    nodes_neg1_df: pd.DataFrame | None,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Phase 2 & Phase 4:
      1. Loads the full PROGENy human resource (`top = progeny_top_genes`).
      2. Identifies active nodes in CARNIVAL `KISS1R = +1` (and optionally `KISS1R = -1`).
      3. Selects the fixed primary PROGENy pathway panel based EXCLUSIVELY on overlap with
         CARNIVAL `KISS1R = +1` (`overlap_count_pos1 >= min_pathway_carnival_overlap`).
      4. Retains the FULL PROGENy gene signature and weights for all selected pathways.
      5. Builds a separate `carnival_subnetwork_net` (`CARNIVAL_Subnetwork_POS1` and
         `CARNIVAL_Subnetwork_NEG1`, excluding `KISS1R` itself when configured) so the
         network footprint is evaluated separately from biological PROGENy pathways.
    """
    top_genes = int(config.get("progeny_top_genes", 500))
    min_overlap = int(config.get("min_pathway_carnival_overlap", 1))
    kiss1r_gene = str(config.get("kiss1r_gene", "KISS1R"))
    excl_kiss_overlap = bool(config.get("exclude_kiss1r_from_pathway_overlap", True))
    excl_kiss_subnet = bool(config.get("exclude_kiss1r_from_subnetwork_score", True))

    progeny_df = dc.op.progeny(organism="human", top=top_genes).copy()
    # Harmonize decoupler column names ('source', 'target', 'weight')
    if "pathway" in progeny_df.columns and "source" not in progeny_df.columns:
        progeny_df = progeny_df.rename(columns={"pathway": "source"})
    if "genesymbol" in progeny_df.columns and "target" not in progeny_df.columns:
        progeny_df = progeny_df.rename(columns={"genesymbol": "target"})

    pos1_sub = nodes_pos1_df.copy()
    if excl_kiss_overlap:
        pos1_sub = pos1_sub[pos1_sub["Node"] != kiss1r_gene].copy()
    pos1_map = dict(zip(pos1_sub["Node"], pos1_sub["Activity"]))

    neg1_map: dict[str, float] = {}
    if nodes_neg1_df is not None and not nodes_neg1_df.empty:
        neg1_sub = nodes_neg1_df.copy()
        if excl_kiss_overlap:
            neg1_sub = neg1_sub[neg1_sub["Node"] != kiss1r_gene].copy()
        neg1_map = dict(zip(neg1_sub["Node"], neg1_sub["Activity"]))

    overlap_rows: list[dict[str, Any]] = []
    for pathway_name, grp in progeny_df.groupby("source"):
        pw_genes = grp.set_index("target")["weight"].to_dict()

        ov_pos1 = sorted(set(pw_genes.keys()) & set(pos1_map.keys()))
        w_score_pos1 = float(sum(pos1_map[g] * pw_genes[g] for g in ov_pos1)) if ov_pos1 else 0.0
        exp_sign_pos1 = float(np.sign(w_score_pos1)) if abs(w_score_pos1) > 1e-9 else 0.0
        sel_pos1 = len(ov_pos1) >= min_overlap

        ov_neg1 = sorted(set(pw_genes.keys()) & set(neg1_map.keys()))
        w_score_neg1 = float(sum(neg1_map[g] * pw_genes[g] for g in ov_neg1)) if ov_neg1 else np.nan
        exp_sign_neg1 = (
            float(np.sign(w_score_neg1))
            if (ov_neg1 and not np.isnan(w_score_neg1) and abs(w_score_neg1) > 1e-9)
            else np.nan
        )
        sel_neg1 = len(ov_neg1) >= min_overlap if neg1_map else False

        if sel_pos1 and sel_neg1:
            cat = "shared"
        elif sel_pos1 and not sel_neg1:
            cat = "pos1_specific"
        elif not sel_pos1 and sel_neg1:
            cat = "neg1_specific"
        else:
            cat = "neither"

        overlap_rows.append(
            {
                "pathway": str(pathway_name),
                "total_progeny_genes": int(len(pw_genes)),
                "overlap_count_pos1": int(len(ov_pos1)),
                "overlapping_genes_pos1": ", ".join(ov_pos1),
                "carnival_pos1_weighted_score": round(w_score_pos1, 4),
                "carnival_pos1_expected_sign": exp_sign_pos1,
                "selected_in_primary_pos1_panel": bool(sel_pos1),
                "overlap_count_neg1": int(len(ov_neg1)) if neg1_map else 0,
                "overlapping_genes_neg1": ", ".join(ov_neg1) if neg1_map else "",
                "carnival_neg1_weighted_score": (
                    round(w_score_neg1, 4) if not np.isnan(w_score_neg1) else np.nan
                ),
                "carnival_neg1_expected_sign": exp_sign_neg1,
                "selected_in_secondary_neg1_panel": bool(sel_neg1),
                "scenario_category": cat,
            }
        )

    overlap_summary_df = pd.DataFrame(overlap_rows).sort_values(
        ["selected_in_primary_pos1_panel", "overlap_count_pos1", "pathway"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    primary_pos1_pathways = overlap_summary_df.loc[
        overlap_summary_df["selected_in_primary_pos1_panel"], "pathway"
    ].tolist()
    secondary_neg1_pathways = overlap_summary_df.loc[
        overlap_summary_df["selected_in_secondary_neg1_panel"], "pathway"
    ].tolist()
    shared_pathways = overlap_summary_df.loc[
        overlap_summary_df["scenario_category"] == "shared", "pathway"
    ].tolist()
    pos1_only_pathways = overlap_summary_df.loc[
        overlap_summary_df["scenario_category"] == "pos1_specific", "pathway"
    ].tolist()
    neg1_only_pathways = overlap_summary_df.loc[
        overlap_summary_df["scenario_category"] == "neg1_specific", "pathway"
    ].tolist()

    # Keep all PROGENy pathways in either +1 or -1 for scoring, but strictly flag the primary +1 panel
    all_selected_progeny = sorted(set(primary_pos1_pathways) | set(secondary_neg1_pathways))
    progeny_panel_net = progeny_df[progeny_df["source"].isin(all_selected_progeny)][
        ["source", "target", "weight"]
    ].copy()

    # Separate CARNIVAL_Subnetwork signature network (excluding KISS1R when configured)
    subnet_rows: list[dict[str, Any]] = []
    for _, row in nodes_pos1_df.iterrows():
        if excl_kiss_subnet and str(row["Node"]) == kiss1r_gene:
            continue
        subnet_rows.append(
            {
                "source": "CARNIVAL_Subnetwork_POS1",
                "target": str(row["Node"]),
                "weight": float(row["Activity"]),
            }
        )
    if nodes_neg1_df is not None and not nodes_neg1_df.empty:
        for _, row in nodes_neg1_df.iterrows():
            if excl_kiss_subnet and str(row["Node"]) == kiss1r_gene:
                continue
            subnet_rows.append(
                {
                    "source": "CARNIVAL_Subnetwork_NEG1",
                    "target": str(row["Node"]),
                    "weight": float(row["Activity"]),
                }
            )
    carnival_subnetwork_net = pd.DataFrame(subnet_rows)

    panel_meta = {
        "progeny_top_genes": top_genes,
        "min_pathway_carnival_overlap": min_overlap,
        "exclude_kiss1r_from_pathway_overlap": excl_kiss_overlap,
        "exclude_kiss1r_from_subnetwork_score": excl_kiss_subnet,
        "primary_pos1_pathways": primary_pos1_pathways,
        "secondary_neg1_pathways": secondary_neg1_pathways,
        "shared_pathways": shared_pathways,
        "pos1_specific_pathways": pos1_only_pathways,
        "neg1_specific_pathways": neg1_only_pathways,
    }
    LOGGER.info(
        "Selected fixed CARNIVAL +1 PROGENy panel (%d pathways): %s | Shared with -1: %s",
        len(primary_pos1_pathways),
        primary_pos1_pathways,
        shared_pathways,
    )
    return progeny_panel_net, carnival_subnetwork_net, overlap_summary_df, panel_meta


def _resolve_inherited_step04_methods(
    cohort_name: str,
    step04_report_path: Path | None,
    config: dict[str, Any],
) -> tuple[list[str], str]:
    """
    Retrieves the exact consensus method combination selected in Step04 for `cohort_name`
    from `step04_report.json` without re-optimizing on Step06 data.
    """
    fallback = [str(m) for m in config["fallback_consensus_methods"]]
    if not bool(config.get("inherit_step04_consensus", True)):
        return fallback, "config_fallback"

    if step04_report_path and step04_report_path.exists():
        try:
            with step04_report_path.open("r", encoding="utf-8") as handle:
                rep04 = json.load(handle)
            # Search inside step04 report structure for the winning combo
            cohorts_dict = rep04.get("cohorts", rep04.get("results", {}))
            for k, val in cohorts_dict.items():
                if k.lower() in {cohort_name.lower(), "pd_male", "pd_males"} and isinstance(val, dict):
                    combo = (
                        val.get("winning_combo")
                        or val.get("optimal_combo")
                        or val.get("selected_methods")
                    )
                    if isinstance(combo, str):
                        methods = [m.strip() for m in combo.replace("+", ",").split(",") if m.strip()]
                        if methods:
                            return methods, f"inherited_from_{step04_report_path.name}"
                    elif isinstance(combo, list) and combo:
                        return [str(m) for m in combo], f"inherited_from_{step04_report_path.name}"
        except Exception as exc:
            LOGGER.warning("Could not parse step04_report.json (%s); using fallback: %s", exc, fallback)

    return fallback, "step04_verified_default_aucell_gsva_zscore"


def _run_decoupler_consensus_on_net(
    pdata_norm: ad.AnnData,
    net_df: pd.DataFrame,
    methods: list[str],
    tmin: int,
) -> pd.DataFrame:
    """
    Runs the inherited Step04 Decoupler methods on `pdata_norm` and `net_df`, computes
    the consensus score matrix (`ASAP_subject_id` x `source`), and guarantees index alignment.
    """
    if net_df.empty:
        return pd.DataFrame(index=pdata_norm.obs_names)

    adata_work = pdata_norm.copy()
    subj_index = pd.Index(adata_work.obs_names.astype(str), name="ASAP_subject_id")

    # Compatible with both decoupler >= 1.8 (dc.mt.decouple) and older dc.decouple
    if hasattr(dc, "mt") and hasattr(dc.mt, "decouple"):
        dc.mt.decouple(
            data=adata_work,
            net=net_df,
            methods=methods,
            tmin=tmin,
            cons=True,
            verbose=False,
        )
        for key in ("score_consensus", "consensus_estimate", "dc_consensus"):
            if key in adata_work.obsm:
                res = adata_work.obsm[key]
                if isinstance(res, pd.DataFrame):
                    return res.reindex(subj_index)
    else:
        dc.decouple(
            mat=adata_work,
            net=net_df,
            methods=methods,
            min_n=tmin,
            consensus=True,
            verbose=False,
        )
        if "consensus_estimate" in adata_work.obsm:
            res = adata_work.obsm["consensus_estimate"]
            if isinstance(res, pd.DataFrame):
                return res.reindex(subj_index)

    # Explicit fallback: mean of z-scored estimates across the requested methods in obsm
    z_mats: list[pd.DataFrame] = []
    for m in methods:
        for obsm_k in (f"score_{m}", f"{m}_estimate", m):
            if obsm_k in adata_work.obsm:
                df_m = adata_work.obsm[obsm_k]
                if isinstance(df_m, pd.DataFrame):
                    df_m = df_m.reindex(subj_index)
                    z_df = (df_m - df_m.mean(axis=0)) / df_m.std(axis=0, ddof=1).replace(0, 1.0)
                    z_mats.append(z_df)
                break

    if not z_mats:
        raise RuntimeError(
            f"Decoupler did not produce score matrices for methods={methods} (available obsm={list(adata_work.obsm.keys())})."
        )
    consensus_df = sum(z_mats) / len(z_mats)
    return consensus_df.reindex(subj_index)


def score_pathways_on_subjects_step04_aware(
    pdata_norm: ad.AnnData,
    subject_df: pd.DataFrame,
    progeny_panel_net: pd.DataFrame,
    carnival_subnetwork_net: pd.DataFrame,
    overlap_summary_df: pd.DataFrame,
    step04_report_path: Path | None,
    config: dict[str, Any],
    cohort_name: str = "pd_male",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Phase 3 & Phase 4:
      1. Scores all male PD subjects simultaneously using the fixed PROGENy pathway signatures
         and the inherited Step04 consensus method combination (`aucell + gsva + zscore`).
      2. Scores `CARNIVAL_Subnetwork_POS1` and `CARNIVAL_Subnetwork_NEG1` separately.
      3. Computes differential statistics (`KISS1R-High` vs `KISS1R-Low`: Welch's t-test,
         95% CI, Cohen's d, Hedges' g, Mann-Whitney U sensitivity test, rank-biserial r,
         and Benjamini-Hochberg FDR across the a priori CARNIVAL `+1` panel).
      4. Computes continuous Spearman and Kendall rank correlations with `kiss1r_log1p_cpm`.
    """
    tmin_pw = int(config.get("tmin_pathway_targets", 5))
    methods, method_source = _resolve_inherited_step04_methods(
        cohort_name=cohort_name,
        step04_report_path=step04_report_path,
        config=config,
    )
    LOGGER.info(
        "Scoring pathways across %d subjects using Step04 consensus methods=%s (%s)...",
        pdata_norm.n_obs,
        methods,
        method_source,
    )

    progeny_scores = _run_decoupler_consensus_on_net(
        pdata_norm=pdata_norm,
        net_df=progeny_panel_net,
        methods=methods,
        tmin=tmin_pw,
    )
    subnet_scores = _run_decoupler_consensus_on_net(
        pdata_norm=pdata_norm,
        net_df=carnival_subnetwork_net,
        methods=methods,
        tmin=min(3, tmin_pw),
    )

    subj_indexed = subject_df.set_index("ASAP_subject_id").reindex(pdata_norm.obs_names.astype(str))
    scores_combined = pd.concat([progeny_scores, subnet_scores], axis=1)

    subject_scores_out = subj_indexed[
        [
            "n_cells",
            "total_counts",
            "kiss1r_raw_counts",
            "kiss1r_log1p_cpm",
            "kiss1r_detected",
            "primary_group",
            "detected_subsplit_group",
        ]
    ].join(scores_combined)

    # Map expected directions from overlap_summary_df
    ov_meta = overlap_summary_df.set_index("pathway")
    high_mask = (subj_indexed["primary_group"] == "KISS1R-High").to_numpy(dtype=bool)
    low_mask = (subj_indexed["primary_group"] == "KISS1R-Low").to_numpy(dtype=bool)
    kiss1r_cont = subj_indexed["kiss1r_log1p_cpm"].to_numpy(dtype=float)
    det_mask = subj_indexed["kiss1r_detected"].to_numpy(dtype=bool)

    diff_rows: list[dict[str, Any]] = []
    corr_rows: list[dict[str, Any]] = []

    for col in scores_combined.columns:
        vals = scores_combined[col].to_numpy(dtype=float)
        x_high = vals[high_mask & ~np.isnan(vals)]
        y_low = vals[low_mask & ~np.isnan(vals)]

        if col in ov_meta.index:
            panel_type = str(ov_meta.loc[col, "scenario_category"])
            in_pos1_panel = bool(ov_meta.loc[col, "selected_in_primary_pos1_panel"])
            exp_sign_pos1 = float(ov_meta.loc[col, "carnival_pos1_expected_sign"])
            exp_sign_neg1 = (
                float(ov_meta.loc[col, "carnival_neg1_expected_sign"])
                if pd.notna(ov_meta.loc[col, "carnival_neg1_expected_sign"])
                else np.nan
            )
        elif col == "CARNIVAL_Subnetwork_POS1":
            panel_type = "carnival_subnetwork_pos1"
            in_pos1_panel = False  # kept separate from PROGENy biological panel FDR
            exp_sign_pos1 = 1.0
            exp_sign_neg1 = np.nan
        elif col == "CARNIVAL_Subnetwork_NEG1":
            panel_type = "carnival_subnetwork_neg1"
            in_pos1_panel = False
            exp_sign_pos1 = np.nan
            exp_sign_neg1 = 1.0
        else:
            panel_type = "other"
            in_pos1_panel = False
            exp_sign_pos1 = np.nan
            exp_sign_neg1 = np.nan

        n1, n2 = len(x_high), len(y_low)
        m1 = float(np.mean(x_high)) if n1 > 0 else np.nan
        m2 = float(np.mean(y_low)) if n2 > 0 else np.nan
        sd1 = float(np.std(x_high, ddof=1)) if n1 > 1 else 0.0
        sd2 = float(np.std(y_low, ddof=1)) if n2 > 1 else 0.0
        med1 = float(np.median(x_high)) if n1 > 0 else np.nan
        med2 = float(np.median(y_low)) if n2 > 0 else np.nan
        iqr1 = float(np.percentile(x_high, 75) - np.percentile(x_high, 25)) if n1 > 0 else 0.0
        iqr2 = float(np.percentile(y_low, 75) - np.percentile(y_low, 25)) if n2 > 0 else 0.0
        delta = m1 - m2

        # Welch's t-test and analytical 95% CI
        se1 = (sd1 ** 2) / n1 if n1 > 0 else np.nan
        se2 = (sd2 ** 2) / n2 if n2 > 0 else np.nan
        se_diff = float(np.sqrt(se1 + se2)) if (n1 > 0 and n2 > 0) else np.nan
        if n1 > 1 and n2 > 1 and se_diff > 0:
            t_stat, p_welch = stats.ttest_ind(x_high, y_low, equal_var=False)
            df_w = float(((se1 + se2) ** 2) / (((se1 ** 2) / (n1 - 1)) + ((se2 ** 2) / (n2 - 1))))
            t_crit = float(stats.t.ppf(0.975, df=df_w))
            ci_low = delta - t_crit * se_diff
            ci_high = delta + t_crit * se_diff
        else:
            t_stat, p_welch, df_w, ci_low, ci_high = 0.0, 1.0, float(max(1, n1 + n2 - 2)), delta, delta

        # Cohen's d and Hedges' g
        df_pool = n1 + n2 - 2
        s_pool = (
            float(np.sqrt(((n1 - 1) * (sd1 ** 2) + (n2 - 1) * (sd2 ** 2)) / df_pool))
            if df_pool > 0
            else 0.0
        )
        if s_pool > 0 and n1 > 0 and n2 > 0:
            cohens_d = delta / s_pool
            j_corr = 1.0 - (3.0 / (4.0 * df_pool - 1.0)) if df_pool > 1 else 1.0
            hedges_g = cohens_d * j_corr
            se_g = float(np.sqrt((n1 + n2) / (n1 * n2) + (hedges_g ** 2) / (2.0 * (n1 + n2))))
            g_ci_low = hedges_g - 1.96 * se_g
            g_ci_high = hedges_g + 1.96 * se_g
        else:
            cohens_d, hedges_g, g_ci_low, g_ci_high = 0.0, 0.0, 0.0, 0.0

        # Mann-Whitney U sensitivity test
        if n1 > 0 and n2 > 0:
            u_stat, p_mw = stats.mannwhitneyu(x_high, y_low, alternative="two-sided")
            r_rb = float(2.0 * (u_stat / (n1 * n2)) - 1.0)
        else:
            u_stat, p_mw, r_rb = np.nan, 1.0, 0.0

        obs_sign_delta = float(np.sign(delta)) if abs(delta) > 1e-9 else 0.0
        concordant_pos1 = (
            bool(obs_sign_delta == exp_sign_pos1)
            if (not np.isnan(exp_sign_pos1) and exp_sign_pos1 != 0.0)
            else None
        )
        concordant_neg1 = (
            bool(obs_sign_delta == exp_sign_neg1)
            if (not np.isnan(exp_sign_neg1) and exp_sign_neg1 != 0.0)
            else None
        )

        diff_rows.append(
            {
                "pathway": col,
                "scenario_category": panel_type,
                "in_primary_pos1_progeny_panel": in_pos1_panel,
                "n_high": n1,
                "n_low": n2,
                "mean_high": round(m1, 4),
                "sd_high": round(sd1, 4),
                "median_high": round(med1, 4),
                "iqr_high": round(iqr1, 4),
                "mean_low": round(m2, 4),
                "sd_low": round(sd2, 4),
                "median_low": round(med2, 4),
                "iqr_low": round(iqr2, 4),
                "delta_activity": round(delta, 4),
                "ci95_low": round(ci_low, 4),
                "ci95_high": round(ci_high, 4),
                "statistical_test_primary": "Welch two-sample t-test",
                "t_stat_welch": round(float(t_stat), 4),
                "df_welch": round(df_w, 2),
                "p_value_welch": float(p_welch),
                "fdr_welch": np.nan,
                "cohens_d": round(float(cohens_d), 4),
                "hedges_g": round(float(hedges_g), 4),
                "hedges_g_ci95_low": round(float(g_ci_low), 4),
                "hedges_g_ci95_high": round(float(g_ci_high), 4),
                "statistical_test_sensitivity": "Mann-Whitney U test",
                "u_stat_mw": round(float(u_stat), 2),
                "p_value_mannwhitney": float(p_mw),
                "fdr_mannwhitney": np.nan,
                "rank_biserial_r": round(r_rb, 4),
                "carnival_pos1_expected_sign": exp_sign_pos1,
                "concordant_with_carnival_pos1": concordant_pos1,
                "carnival_neg1_expected_sign": exp_sign_neg1,
                "concordant_with_carnival_neg1": concordant_neg1,
            }
        )

        # Continuous Spearman & Kendall correlation across all subjects
        valid_c = ~np.isnan(vals) & ~np.isnan(kiss1r_cont)
        n_c = int(np.sum(valid_c))
        if n_c >= 4 and np.std(vals[valid_c]) > 0 and np.std(kiss1r_cont[valid_c]) > 0:
            rho, p_sp = stats.spearmanr(kiss1r_cont[valid_c], vals[valid_c])
            tau, p_kd = stats.kendalltau(kiss1r_cont[valid_c], vals[valid_c])
            # Fisher z 95% CI for Spearman rho
            if abs(rho) < 0.9999 and n_c > 3:
                z = np.arctanh(rho)
                se_z = 1.0 / np.sqrt(n_c - 3)
                rho_low, rho_high = float(np.tanh(z - 1.96 * se_z)), float(np.tanh(z + 1.96 * se_z))
            else:
                rho_low, rho_high = float(rho), float(rho)
        else:
            rho, p_sp, tau, p_kd, rho_low, rho_high = 0.0, 1.0, 0.0, 1.0, 0.0, 0.0

        # Detected-only Spearman correlation (exploratory)
        valid_det = valid_c & det_mask
        if int(np.sum(valid_det)) >= 4 and np.std(vals[valid_det]) > 0:
            rho_det, p_det = stats.spearmanr(kiss1r_cont[valid_det], vals[valid_det])
        else:
            rho_det, p_det = np.nan, np.nan

        obs_sign_rho = float(np.sign(rho)) if abs(rho) > 1e-9 else 0.0
        corr_concordant_pos1 = (
            bool(obs_sign_rho == exp_sign_pos1)
            if (not np.isnan(exp_sign_pos1) and exp_sign_pos1 != 0.0)
            else None
        )

        corr_rows.append(
            {
                "pathway": col,
                "scenario_category": panel_type,
                "in_primary_pos1_progeny_panel": in_pos1_panel,
                "n_subjects": n_c,
                "spearman_rho": round(float(rho), 4),
                "spearman_ci95_low": round(rho_low, 4),
                "spearman_ci95_high": round(rho_high, 4),
                "p_value_spearman": float(p_sp),
                "fdr_spearman": np.nan,
                "kendall_tau": round(float(tau), 4),
                "p_value_kendall": float(p_kd),
                "n_detected_subjects": int(np.sum(valid_det)),
                "spearman_rho_detected_only": (
                    round(float(rho_det), 4) if not np.isnan(rho_det) else np.nan
                ),
                "p_value_detected_only": float(p_det) if not np.isnan(p_det) else np.nan,
                "carnival_pos1_expected_sign": exp_sign_pos1,
                "concordant_with_carnival_pos1": corr_concordant_pos1,
            }
        )

    diff_df = pd.DataFrame(diff_rows)
    corr_df = pd.DataFrame(corr_rows)

    # Apply Benjamini-Hochberg FDR strictly across the a priori CARNIVAL +1 PROGENy panel
    pos1_panel_mask = diff_df["in_primary_pos1_progeny_panel"].to_numpy(dtype=bool)
    if np.any(pos1_panel_mask):
        diff_df.loc[pos1_panel_mask, "fdr_welch"] = _benjamini_hochberg_fdr(
            diff_df.loc[pos1_panel_mask, "p_value_welch"]
        )
        diff_df.loc[pos1_panel_mask, "fdr_mannwhitney"] = _benjamini_hochberg_fdr(
            diff_df.loc[pos1_panel_mask, "p_value_mannwhitney"]
        )
        corr_df.loc[pos1_panel_mask, "fdr_spearman"] = _benjamini_hochberg_fdr(
            corr_df.loc[pos1_panel_mask, "p_value_spearman"]
        )

    # Also compute secondary FDR across the -1 panel for secondary inspection
    neg1_panel_mask = diff_df["scenario_category"].isin(["shared", "neg1_specific"]).to_numpy(dtype=bool)
    if np.any(neg1_panel_mask):
        diff_df["fdr_welch_neg1_panel"] = np.nan
        diff_df.loc[neg1_panel_mask, "fdr_welch_neg1_panel"] = _benjamini_hochberg_fdr(
            diff_df.loc[neg1_panel_mask, "p_value_welch"]
        )

    scoring_meta = {
        "inherited_consensus_methods": methods,
        "consensus_methods_source": method_source,
        "tmin_pathway_targets": tmin_pw,
        "n_primary_pos1_pathways_scored": int(np.sum(pos1_panel_mask)),
    }
    return subject_scores_out.reset_index(), diff_df, corr_df, scoring_meta


def evaluate_carnival_nodes_and_edges_on_subjects(
    pdata_norm: ad.AnnData,
    subject_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    scenario_label: str,
    kiss1r_gene: str = "KISS1R",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Phase 5:
      Exploratory transcriptomic evaluation of CARNIVAL nodes and edges across subject
      pseudobulk profiles. Explicitly flags `KISS1R` (`is_grouping_variable = True`) and
      EXCLUDES `KISS1R` from all aggregate node concordance statistics so the grouping
      variable never inflates validation metrics.
    """
    subj_indexed = subject_df.set_index("ASAP_subject_id").reindex(pdata_norm.obs_names.astype(str))
    high_mask = (subj_indexed["primary_group"] == "KISS1R-High").to_numpy(dtype=bool)
    low_mask = (subj_indexed["primary_group"] == "KISS1R-Low").to_numpy(dtype=bool)

    node_rows: list[dict[str, Any]] = []
    for _, row in nodes_df.iterrows():
        gene = str(row["Node"])
        pred_act = float(row["Activity"])
        pred_sign = float(np.sign(pred_act))
        is_grouping = gene.upper() == kiss1r_gene.upper()

        expr = _extract_1d_expr(pdata_norm, gene)
        present = gene in pdata_norm.var_names
        m_high = float(np.mean(expr[high_mask])) if np.any(high_mask) else np.nan
        m_low = float(np.mean(expr[low_mask])) if np.any(low_mask) else np.nan
        delta = m_high - m_low

        if present and np.std(expr[high_mask]) + np.std(expr[low_mask]) > 0:
            t_val, p_val = stats.ttest_ind(expr[high_mask], expr[low_mask], equal_var=False)
            _, p_mw = stats.mannwhitneyu(expr[high_mask], expr[low_mask], alternative="two-sided")
        else:
            t_val, p_val, p_mw = 0.0, 1.0, 1.0

        obs_sign = float(np.sign(delta)) if abs(delta) > 1e-9 else 0.0
        concordant = None if is_grouping else bool(obs_sign == pred_sign)

        node_rows.append(
            {
                "scenario": scenario_label,
                "node": gene,
                "is_grouping_variable": is_grouping,
                "included_in_aggregate_concordance": not is_grouping and present,
                "present_in_pseudobulk": present,
                "carnival_predicted_activity": pred_act,
                "carnival_predicted_sign": pred_sign,
                "mean_expr_high": round(m_high, 4),
                "mean_expr_low": round(m_low, 4),
                "delta_expr_high_minus_low": round(delta, 4),
                "t_stat_welch": round(float(t_val), 4),
                "p_value_welch": float(p_val),
                "p_value_mannwhitney": float(p_mw),
                "mrna_direction_concordant": concordant,
                "interpretation_note": (
                    "Grouping variable (excluded from validation concordance)"
                    if is_grouping
                    else "Exploratory mRNA abundance check (does not measure kinase/protein activity)"
                ),
            }
        )

    node_eval_df = pd.DataFrame(node_rows)
    indep_mask = node_eval_df["included_in_aggregate_concordance"].to_numpy(dtype=bool)
    if np.any(indep_mask):
        node_eval_df.loc[indep_mask, "fdr_welch"] = _benjamini_hochberg_fdr(
            node_eval_df.loc[indep_mask, "p_value_welch"]
        )

    edge_rows: list[dict[str, Any]] = []
    for _, row in edges_df.iterrows():
        src = str(row["Source"])
        tgt = str(row["Target"])
        inter = float(row["Interaction"])
        involves_kiss = (src.upper() == kiss1r_gene.upper()) or (tgt.upper() == kiss1r_gene.upper())

        expr_s = _extract_1d_expr(pdata_norm, src)
        expr_t = _extract_1d_expr(pdata_norm, tgt)
        if (src in pdata_norm.var_names) and (tgt in pdata_norm.var_names) and np.std(expr_s) > 0 and np.std(expr_t) > 0:
            rho, p_sp = stats.spearmanr(expr_s, expr_t)
        else:
            rho, p_sp = 0.0, 1.0

        obs_corr_sign = float(np.sign(rho)) if abs(rho) > 1e-9 else 0.0
        sign_match = bool(obs_corr_sign == float(np.sign(inter)))

        edge_rows.append(
            {
                "scenario": scenario_label,
                "source": src,
                "interaction_sign": inter,
                "target": tgt,
                "involves_grouping_variable": involves_kiss,
                "spearman_rho_mrna": round(float(rho), 4),
                "p_value_spearman": float(p_sp),
                "mrna_coexpression_matches_interaction_sign": sign_match,
                "interpretation_note": (
                    "Involves grouping variable KISS1R"
                    if involves_kiss
                    else "Exploratory mRNA co-expression (does not prove causal post-translational interaction)"
                ),
            }
        )

    edge_eval_df = pd.DataFrame(edge_rows)
    if not edge_eval_df.empty:
        edge_eval_df["fdr_spearman"] = _benjamini_hochberg_fdr(edge_eval_df["p_value_spearman"])

    indep_nodes = node_eval_df[node_eval_df["included_in_aggregate_concordance"]]
    n_indep = len(indep_nodes)
    n_conc_nodes = int(indep_nodes["mrna_direction_concordant"].sum()) if n_indep > 0 else 0

    downstream_edges = edge_eval_df[~edge_eval_df["involves_grouping_variable"]] if not edge_eval_df.empty else edge_eval_df
    n_ds_edges = len(downstream_edges)
    n_conc_edges = (
        int(downstream_edges["mrna_coexpression_matches_interaction_sign"].sum())
        if n_ds_edges > 0
        else 0
    )

    summary = {
        "scenario": scenario_label,
        "total_nodes_in_network": int(len(node_eval_df)),
        "grouping_variable_excluded": kiss1r_gene,
        "independent_downstream_nodes_evaluated": int(n_indep),
        "concordant_downstream_nodes_count": n_conc_nodes,
        "downstream_node_concordance_rate": round(n_conc_nodes / n_indep, 4) if n_indep > 0 else None,
        "total_edges_in_network": int(len(edge_eval_df)),
        "downstream_edges_excluding_kiss1r": int(n_ds_edges),
        "concordant_downstream_edges_count": n_conc_edges,
        "downstream_edge_concordance_rate": (
            round(n_conc_edges / n_ds_edges, 4) if n_ds_edges > 0 else None
        ),
    }
    return node_eval_df, edge_eval_df, summary


def save_step06_plots(
    subject_df: pd.DataFrame,
    subject_scores_df: pd.DataFrame,
    diff_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    plots_dir: Path,
    cohort_name: str,
    config: dict[str, Any],
) -> dict[str, str]:
    """
    Generates the 4 required diagnostic and validation figures:
      1. Distribution of subject-level pseudobulk KISS1R expression.
      2. Heatmap of pathway activities across subjects ordered by KISS1R expression.
      3. Forest/bar plot of Delta Activity (High - Low) with 95% CIs and CARNIVAL +1 concordance.
      4. Scatterplots of continuous KISS1R expression vs primary CARNIVAL +1 pathways.
    """
    dpi = int(config.get("dpi_figures", 300))
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    tag = cohort_name.lower()

    # 1. KISS1R pseudobulk distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), facecolor="white")
    df_ord = subject_df.sort_values("kiss1r_log1p_cpm", ascending=True).reset_index(drop=True)
    palette_grp = {"KISS1R-High": "#d95f02", "KISS1R-Low": "#1f78b4"}

    sns.histplot(
        data=df_ord,
        x="kiss1r_log1p_cpm",
        hue="primary_group",
        palette=palette_grp,
        bins=10,
        kde=False,
        ax=axes[0],
        alpha=0.8,
        edgecolor="black",
    )
    thresh = float(df_ord["classification_threshold"].iloc[0])
    axes[0].axvline(thresh, color="black", linestyle="--", lw=1.5, label=f"Threshold ({thresh:.3f})")
    axes[0].set_title(f"A. Distribuzione Pseudobulk KISS1R ({cohort_name.upper()}, N={len(df_ord)})", fontweight="bold")
    axes[0].set_xlabel("Espressione Pseudobulk KISS1R [log1p(CPM_10k)]")
    axes[0].set_ylabel("Numero di Pazienti (ASAP_subject_id)")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    bar_colors = [palette_grp.get(g, "#7f7f7f") for g in df_ord["primary_group"]]
    axes[1].barh(df_ord["ASAP_subject_id"].astype(str), df_ord["kiss1r_log1p_cpm"], color=bar_colors, edgecolor="black")
    axes[1].set_title("B. Profilo per Soggetto Ordinato per Espressione di KISS1R", fontweight="bold")
    axes[1].set_xlabel("Espressione Pseudobulk KISS1R [log1p(CPM_10k)]")
    axes[1].set_ylabel("ASAP_subject_id")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.tight_layout()
    p_dist = plots_dir / f"step06_kiss1r_pseudobulk_distribution_{tag}.png"
    fig.savefig(p_dist, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    saved["kiss1r_distribution_plot"] = str(p_dist)

    # 2. Pathway Heatmap across subjects ordered by KISS1R expression
    pos1_pws = diff_df.loc[diff_df["in_primary_pos1_progeny_panel"], "pathway"].tolist()
    extra_cols = [c for c in ["CARNIVAL_Subnetwork_POS1", "CARNIVAL_Subnetwork_NEG1"] if c in subject_scores_df.columns]
    hm_cols = pos1_pws + extra_cols
    if hm_cols:
        scores_ord = subject_scores_df.sort_values(["kiss1r_log1p_cpm", "primary_group"], ascending=[True, True]).set_index("ASAP_subject_id")
        mat = scores_ord[hm_cols].T.copy()
        mat_z = mat.sub(mat.mean(axis=1), axis=0).div(mat.std(axis=1, ddof=1).replace(0, 1.0), axis=0)

        fig, ax = plt.subplots(figsize=(max(8.5, len(scores_ord) * 0.62), max(4.5, len(hm_cols) * 0.48)), facecolor="white")
        sns.heatmap(
            mat_z,
            cmap="RdBu_r",
            center=0.0,
            annot=True,
            fmt=".2f",
            linewidths=0.6,
            cbar_kws={"label": "Z-score Attività Consenso per Soggetto"},
            ax=ax,
        )
        x_labels = [
            f"{sid}\n({'H' if grp == 'KISS1R-High' else 'L'}: {val:.2f})"
            for sid, grp, val in zip(scores_ord.index, scores_ord["primary_group"], scores_ord["kiss1r_log1p_cpm"])
        ]
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8.5)
        ax.set_title(
            f"Attività dei Pathway (Pannello CARNIVAL +1) Ordinata per Espressione Endogena di KISS1R ({cohort_name.upper()})",
            fontweight="bold",
            pad=12,
        )
        ax.set_xlabel("Soggetto (Gruppo KISS1R: valore log1p_CPM)")
        ax.set_ylabel("Pathway PROGENy / Firma CARNIVAL")
        fig.tight_layout()
        p_hm = plots_dir / f"step06_pathway_heatmap_ordered_by_kiss1r_{tag}.png"
        fig.savefig(p_hm, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved["pathway_heatmap_plot"] = str(p_hm)

    # 3. Delta Activity (High - Low) forest/bar plot with 95% CIs
    plot_diff = diff_df[diff_df["in_primary_pos1_progeny_panel"] | diff_df["pathway"].str.startswith("CARNIVAL_Subnetwork")].copy()
    if not plot_diff.empty:
        plot_diff = plot_diff.sort_values("delta_activity", ascending=True).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(9.5, max(4.5, len(plot_diff) * 0.52)), facecolor="white")
        y_pos = np.arange(len(plot_diff))
        colors = [
            "#2ca02c" if c is True else ("#d62728" if c is False else "#7f7f7f")
            for c in plot_diff["concordant_with_carnival_pos1"]
        ]
        ax.barh(y_pos, plot_diff["delta_activity"], color=colors, edgecolor="black", alpha=0.85, height=0.6)
        xerr_low = plot_diff["delta_activity"] - plot_diff["ci95_low"]
        xerr_high = plot_diff["ci95_high"] - plot_diff["delta_activity"]
        ax.errorbar(
            plot_diff["delta_activity"],
            y_pos,
            xerr=[xerr_low, xerr_high],
            fmt="none",
            ecolor="black",
            capsize=3,
            lw=1.2,
        )
        ax.axvline(0.0, color="black", linestyle="--", lw=1.2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_diff["pathway"], fontweight="bold")
        ax.set_xlabel("Delta Activity (KISS1R-High minus KISS1R-Low) ± 95% Welch CI")
        ax.set_title(
            f"Differenza di Attività dei Pathway (KISS1R-High vs KISS1R-Low, {cohort_name.upper()})\n"
            "[Verde = Concordante con CARNIVAL +1 | Rosso = Discordante]",
            fontweight="bold",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        p_delta = plots_dir / f"step06_pathway_delta_activity_high_vs_low_{tag}.png"
        fig.savefig(p_delta, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved["pathway_delta_plot"] = str(p_delta)

    # 4. Continuous KISS1R expression vs top CARNIVAL +1 pathways scatterplots
    top_scatter_pws = (
        corr_df[corr_df["in_primary_pos1_progeny_panel"] | (corr_df["pathway"] == "CARNIVAL_Subnetwork_POS1")]
        .sort_values("p_value_spearman", ascending=True)["pathway"]
        .head(6)
        .tolist()
    )
    if top_scatter_pws:
        n_panels = len(top_scatter_pws)
        ncols = min(3, n_panels)
        nrows = (n_panels + ncols - 1) // ncols
        fig, axes_grid = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows), facecolor="white", squeeze=False)
        corr_indexed = corr_df.set_index("pathway")

        for idx, pw in enumerate(top_scatter_pws):
            ax = axes_grid[idx // ncols][idx % ncols]
            sns.regplot(
                data=subject_scores_df,
                x="kiss1r_log1p_cpm",
                y=pw,
                ax=ax,
                scatter_kws={"s": 45, "edgecolor": "black", "alpha": 0.85},
                line_kws={"color": "#d95f02", "lw": 1.8},
            )
            rho_v = corr_indexed.loc[pw, "spearman_rho"]
            p_v = corr_indexed.loc[pw, "p_value_spearman"]
            fdr_v = corr_indexed.loc[pw, "fdr_spearman"]
            fdr_str = f", FDR={fdr_v:.3f}" if pd.notna(fdr_v) else ""
            ax.set_title(f"{pw}\nSpearman ρ={rho_v:.2f} (p={p_v:.3f}{fdr_str})", fontsize=9.5, fontweight="bold")
            ax.set_xlabel("KISS1R [log1p(CPM_10k)]")
            ax.set_ylabel("Pathway Activity Score")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        for idx in range(n_panels, nrows * ncols):
            axes_grid[idx // ncols][idx % ncols].axis("off")

        fig.tight_layout()
        p_scat = plots_dir / f"step06_kiss1r_continuous_scatterplots_{tag}.png"
        fig.savefig(p_scat, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved["continuous_scatter_plot"] = str(p_scat)

    return saved


def build_step06_parser(default_repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Step06 CARNIVAL-Based Pathway, Node, and Edge Validation in "
            "KISS1R-High vs KISS1R-Low PD Patients"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    processed_dir = default_repo_root / "pipeline" / "data" / "processed"
    plots_dir = default_repo_root / "pipeline" / "plots"
    reports_dir = default_repo_root / "pipeline" / "reports"

    parser.add_argument(
        "--config",
        default=str(default_repo_root / "pipeline" / "config" / "step06.yml"),
        help="YAML config file for step06.",
    )
    parser.add_argument(
        "--step04-config",
        default=str(default_repo_root / "pipeline" / "config" / "step04.yml"),
        help="YAML config file from step04 to synchronize pseudobulk parameters.",
    )
    parser.add_argument(
        "--step04-report",
        default=str(reports_dir / "step04_report.json"),
        help="Path to step04_report.json to inherit the winning Decoupler consensus combination.",
    )
    parser.add_argument(
        "--step05-report",
        default=str(reports_dir / "step05_report.json"),
        help="Path to step05_report.json.",
    )
    parser.add_argument(
        "--input-h5ad",
        default=str(processed_dir / "dataset_M_Dopa_filtered.h5ad"),
        help="Path to the filtered dopaminergic AnnData (.h5ad) used in Step04.",
    )
    parser.add_argument(
        "--carnival-pos1-csv",
        default=str(processed_dir / "step05_carnival" / "carnival_network_KISS1R_pos1.csv"),
        help="Path to the CARNIVAL network CSV for scenario KISS1R = +1.",
    )
    parser.add_argument(
        "--carnival-neg1-csv",
        default=str(processed_dir / "step05_carnival" / "carnival_network_KISS1R_neg1.csv"),
        help="Optional path to the CARNIVAL network CSV for secondary scenario KISS1R = -1.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(processed_dir / "step06_validation"),
        help="Directory where Step06 validation CSVs will be saved.",
    )
    parser.add_argument("--plots-dir", default=str(plots_dir))
    parser.add_argument("--reports-dir", default=str(reports_dir))
    parser.add_argument("--logging-level", default="INFO")
    return parser


def run_step06(args: argparse.Namespace) -> None:
    configure_logging(args.logging_level)
    config_path = Path(args.config)
    s04_cfg_path = Path(args.step04_config) if args.step04_config else None
    s04_rep_path = Path(args.step04_report) if args.step04_report else None
    config = _resolve_step06_config(config_path, s04_cfg_path)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    h5ad_path = Path(args.input_h5ad)
    if not h5ad_path.exists():
        fallback_h5ad = h5ad_path.parent / CANONICAL_NAMES.get("dataset_dopaminergic", "dataset_dopaminergic.h5ad")
        if fallback_h5ad.exists():
            h5ad_path = fallback_h5ad
        else:
            raise FileNotFoundError(f"Dopaminergic input h5ad not found: {h5ad_path}")

    s05_rep_path = Path(args.step05_report) if args.step05_report else None
    primary_cohort = config["cohorts"][0] if config["cohorts"] else "pd_male"

    pos1_csv, neg1_csv = _resolve_carnival_csv_paths(
        pos1_arg=args.carnival_pos1_csv,
        neg1_arg=args.carnival_neg1_csv,
        step05_report_path=s05_rep_path,
        processed_dir=Path(args.output_dir).parent,
        cohort_name=primary_cohort,
    )

    LOGGER.info("Loading CARNIVAL KISS1R = +1 network from %s...", pos1_csv)
    nodes_pos1, edges_pos1 = _parse_carnival_network_csv(pos1_csv)

    nodes_neg1, edges_neg1 = (None, None)
    if neg1_csv and neg1_csv.exists():
        LOGGER.info("Loading secondary CARNIVAL KISS1R = -1 network from %s...", neg1_csv)
        nodes_neg1, edges_neg1 = _parse_carnival_network_csv(neg1_csv)

    # Phase 2: Fixed a priori pathway selection based on CARNIVAL KISS1R = +1
    progeny_panel_net, carnival_subnetwork_net, overlap_summary_df, panel_meta = (
        build_pathway_network_for_carnival(
            nodes_pos1_df=nodes_pos1,
            nodes_neg1_df=nodes_neg1,
            config=config,
        )
    )

    LOGGER.info("Loading dopaminergic dataset: %s...", h5ad_path)
    adata_dopa = sc.read_h5ad(h5ad_path)

    cohort_reports: dict[str, Any] = {}
    for cohort_name in config["cohorts"]:
        tag = cohort_name.lower()
        adata_cohort = _filter_cohort_cells(adata_dopa, cohort_name)
        LOGGER.info("=== Step06 Validation for Cohort: %s (cells=%d) ===", cohort_name.upper(), adata_cohort.n_obs)

        # Phase 1: Subject pseudobulk & statistically sound KISS1R stratification
        pdata_raw, pdata_norm, subject_df, strat_summary = prepare_subject_pseudobulk(
            adata_cohort=adata_cohort,
            config=config,
        )

        # Phase 3 & 4: Pathway scoring (using full PROGENy signatures & Step04 consensus)
        subject_scores_df, diff_df, corr_df, scoring_meta = score_pathways_on_subjects_step04_aware(
            pdata_norm=pdata_norm,
            subject_df=subject_df,
            progeny_panel_net=progeny_panel_net,
            carnival_subnetwork_net=carnival_subnetwork_net,
            overlap_summary_df=overlap_summary_df,
            step04_report_path=s04_rep_path,
            config=config,
            cohort_name=cohort_name,
        )

        # Phase 5: Exploratory node and edge evaluation (excluding KISS1R from aggregate concordance)
        nodes_eval_pos1, edges_eval_pos1, ne_summary_pos1 = evaluate_carnival_nodes_and_edges_on_subjects(
            pdata_norm=pdata_norm,
            subject_df=subject_df,
            nodes_df=nodes_pos1,
            edges_df=edges_pos1,
            scenario_label="KISS1R_pos1",
            kiss1r_gene=str(config["kiss1r_gene"]),
        )
        nodes_eval_list = [nodes_eval_pos1]
        edges_eval_list = [edges_eval_pos1]
        ne_summaries = {"KISS1R_pos1": ne_summary_pos1}

        if nodes_neg1 is not None and edges_neg1 is not None:
            nodes_eval_neg1, edges_eval_neg1, ne_summary_neg1 = evaluate_carnival_nodes_and_edges_on_subjects(
                pdata_norm=pdata_norm,
                subject_df=subject_df,
                nodes_df=nodes_neg1,
                edges_df=edges_neg1,
                scenario_label="KISS1R_neg1",
                kiss1r_gene=str(config["kiss1r_gene"]),
            )
            nodes_eval_list.append(nodes_eval_neg1)
            edges_eval_list.append(edges_eval_neg1)
            ne_summaries["KISS1R_neg1"] = ne_summary_neg1

        nodes_eval_all = pd.concat(nodes_eval_list, ignore_index=True)
        edges_eval_all = pd.concat(edges_eval_list, ignore_index=True)

        # Save all 6 required CSV outputs
        csv_paths = {
            "subject_kiss1r_stratification_csv": str(out_dir / f"step06_subject_kiss1r_stratification_{tag}.csv"),
            "subject_pathway_scores_csv": str(out_dir / f"step06_subject_pathway_scores_{tag}.csv"),
            "differential_pathways_high_vs_low_csv": str(out_dir / f"step06_differential_pathways_high_vs_low_{tag}.csv"),
            "continuous_kiss1r_pathway_correlations_csv": str(out_dir / f"step06_continuous_kiss1r_pathway_correlations_{tag}.csv"),
            "carnival_progeny_overlap_summary_csv": str(out_dir / f"step06_carnival_progeny_overlap_summary_{tag}.csv"),
            "carnival_nodes_validation_csv": str(out_dir / f"step06_carnival_nodes_validation_{tag}.csv"),
            "carnival_edges_validation_csv": str(out_dir / f"step06_carnival_edges_validation_{tag}.csv"),
        }
        subject_df.to_csv(csv_paths["subject_kiss1r_stratification_csv"], index=False)
        subject_scores_df.to_csv(csv_paths["subject_pathway_scores_csv"], index=False)
        diff_df.to_csv(csv_paths["differential_pathways_high_vs_low_csv"], index=False)
        corr_df.to_csv(csv_paths["continuous_kiss1r_pathway_correlations_csv"], index=False)
        overlap_summary_df.to_csv(csv_paths["carnival_progeny_overlap_summary_csv"], index=False)
        nodes_eval_all.to_csv(csv_paths["carnival_nodes_validation_csv"], index=False)
        edges_eval_all.to_csv(csv_paths["carnival_edges_validation_csv"], index=False)

        # Save all 4 required diagnostic and validation plots
        saved_plots = save_step06_plots(
            subject_df=subject_df,
            subject_scores_df=subject_scores_df,
            diff_df=diff_df,
            corr_df=corr_df,
            plots_dir=plots_dir,
            cohort_name=cohort_name,
            config=config,
        )

        fdr_cut = float(config.get("fdr_threshold", 0.05))
        sig_diff = diff_df[
            diff_df["in_primary_pos1_progeny_panel"] & (diff_df["fdr_welch"] < fdr_cut)
        ]["pathway"].tolist()
        sig_corr = corr_df[
            corr_df["in_primary_pos1_progeny_panel"] & (corr_df["fdr_spearman"] < fdr_cut)
        ]["pathway"].tolist()

        cohort_reports[cohort_name] = {
            "kiss1r_stratification": strat_summary,
            "pathway_panel_selection": panel_meta,
            "scoring_metadata": scoring_meta,
            "significant_pathways_high_vs_low_fdr": sig_diff,
            "significant_pathways_continuous_spearman_fdr": sig_corr,
            "nodes_and_edges_exploratory_summary": ne_summaries,
            "output_csvs": csv_paths,
            "plots": saved_plots,
        }

    report = {
        "step": "step06_carnival_validation",
        "validation_nature": (
            "INTERNAL_CONCORDANCE_ASSESSMENT: Because the downstream TF constraint (REST) "
            "used by CARNIVAL in Step05 was inferred from the same 15 male PD pseudobulk "
            "profiles in Step04, agreement in Step06 evaluates internal pathway/transcriptomic "
            "consistency rather than independent external cohort validation."
        ),
        "methodological_limitations": [
            "Endogenous KISS1R expression in post-mortem dopaminergic pseudobulk profiles is sparse/zero-inflated (6/15 detected vs 9/15 undetected at min_cells=30).",
            "mRNA abundance of signaling intermediates (e.g. GNAQ, HRAS, MAPK1, ABL1, PLK1) does not directly measure phosphorylation state or kinase catalytic flux.",
            "PROGENy pathway scores measure transcriptional footprints downstream of signaling cascades, which provides a more appropriate readout than raw node mRNA levels.",
            "KISS1R itself is excluded from aggregate node concordance and CARNIVAL_Subnetwork target genes to prevent circularity with the grouping variable.",
        ],
        "config_path": str(config_path),
        "input_h5ad": str(h5ad_path),
        "carnival_pos1_csv": str(pos1_csv),
        "carnival_neg1_csv": str(neg1_csv) if neg1_csv else None,
        "parameters": config,
        "cohorts": cohort_reports,
    }

    report_path = reports_dir / "step06_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    LOGGER.info("Step06 completed successfully. Report saved to: %s", report_path)