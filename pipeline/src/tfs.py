from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path
from typing import Any

import anndata as ad
import decoupler as dc
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
from statsmodels.stats.multitest import multipletests

from pipeline.src.schemas import CANONICAL_NAMES
from pipeline.src.utils import (
    _ensure_list,
    _extract_1d_expr,
    build_tagged_path,
    configure_logging,
    load_yaml_file,
)

LOGGER = logging.getLogger(__name__)

VALID_COHORTS = (
    "control_male",
    "control_female",
    "pd_male",
    "pd_female",
    "male",
    "female",
    "all",
)

DEFAULT_STEP04_CONFIG: dict[str, Any] = {
    "min_cells_pseudobulk": 20,
    "target_sum_norm": 10000.0,
    "stratification_mode": "proxy",  # "proxy" or "kiss1r"
    "kiss1r_gene": "KISS1R",
    "gnaq_gene": "GNAQ",
    "proxy_genes": [
        "GNA11",
        "GRK2",
        "ARRB1",
        "ARRB2",
        "PLCB1",
        "PLCB2",
        "PLCB3",
        "PLCB4",
        "ITPR1",
        "ITPR2",
        "ITPR3",
        "PRKCA",
        "PRKCB",
        "TRPC1",
        "TRPC3",
        "TRPC4",
        "TRPC5",
        "TRPC6",
        "TRPC7",
        "MAPK1",
        "MAPK3",
        "MAPK14",
    ],
    "min_active_proxies": 20,
    "stability_proxy_thresholds": [18, 19, 20, 21],
    "dorothea_levels": ["A", "B", "C"],
    "decoupler_methods": [
        "ulm",
        "waggr",
        "viper",
        "aucell",
        "gsea",
        "gsva",
        "mlm",
        "zscore",
    ],
    "primary_consensus_methods": ["viper", "gsea", "mlm"],
    "run_stability_audit": True,
    "min_combo_size": 3,
    "tmin_targets": 10,
    "fdr_threshold": 0.10,
    "nominal_p_threshold": 0.05,
    "diff_threshold": 0.25,
}


def _resolve_step04_config(config_path: Path) -> dict[str, Any]:
    config = DEFAULT_STEP04_CONFIG.copy()
    if config_path.exists():
        loaded = load_yaml_file(config_path) or {}
        step04_cfg = loaded.get("step04", {})
        if step04_cfg and not isinstance(step04_cfg, dict):
            raise ValueError(f"Expected mapping under 'step04' in config: {config_path}")
        config.update({k: v for k, v in step04_cfg.items() if v is not None})
    else:
        LOGGER.warning("Config file not found (%s), using default values.", config_path)

    config["proxy_genes"] = [str(g) for g in _ensure_list(config["proxy_genes"])]
    config["dorothea_levels"] = [str(lvl) for lvl in _ensure_list(config["dorothea_levels"])]
    config["decoupler_methods"] = [str(m) for m in _ensure_list(config["decoupler_methods"])]

    raw_primary = _ensure_list(config.get("primary_consensus_methods", ["viper", "gsea", "mlm"]))
    if len(raw_primary) == 1 and str(raw_primary[0]).lower() == "all":
        config["primary_consensus_methods"] = list(config["decoupler_methods"])
    else:
        config["primary_consensus_methods"] = [str(m) for m in raw_primary]

    raw_thresh = _ensure_list(config.get("stability_proxy_thresholds", []))
    config["stability_proxy_thresholds"] = [int(x) for x in raw_thresh]
    config["run_stability_audit"] = bool(config.get("run_stability_audit", True))
    return config


def _subset_adata_by_cohort(adata: ad.AnnData, cohort_key: str) -> tuple[str, ad.AnnData]:
    """Filters AnnData according to the requested cohort key."""
    sex_s = adata.obs["sex"].astype(str)
    cond_s = adata.obs["condition_id"].astype(str)

    cohort_map: dict[str, tuple[str, pd.Series]] = {
        "control_male": ("CONTROL_MALES", (sex_s == "Male") & (cond_s == "Control")),
        "control_female": ("CONTROL_FEMALES", (sex_s == "Female") & (cond_s == "Control")),
        "pd_male": ("PD_MALES", (sex_s == "Male") & (cond_s == "PD")),
        "pd_female": ("PD_FEMALES", (sex_s == "Female") & (cond_s == "PD")),
        "male": ("MALES", sex_s == "Male"),
        "female": ("FEMALES", sex_s == "Female"),
    }

    if cohort_key not in cohort_map:
        raise ValueError(f"Unsupported cohort key '{cohort_key}'.")

    label, mask = cohort_map[cohort_key]
    return label, adata[mask].copy()


def make_pseudobulk_stratified(
    adata: ad.AnnData,
    cohort_label: str,
    config: dict[str, Any],
    min_cells: int = 20,
    stratification_mode: str = "proxy",
) -> tuple[ad.AnnData, dict[str, Any]]:
    """
    Generates subject-level pseudobulk profiles, filters low-cell samples,
    and stratifies subjects into 'High' vs 'Low' using either KISS1R + proxies or KISS1R alone.
    Also records alternative stratification columns in .obs for cross-configuration stability auditing.
    """
    LOGGER.info(
        "Generating pseudobulk for %s (min_cells=%d, stratification=%s)...",
        cohort_label,
        min_cells,
        stratification_mode,
    )

    kiss1r_gene = str(config["kiss1r_gene"])
    gnaq_gene = str(config["gnaq_gene"])
    proxy_genes: list[str] = config["proxy_genes"]
    min_proxies = int(config["min_active_proxies"])
    target_sum = float(config["target_sum_norm"])

    # Validate required genes in adata.var_names
    required_genes = [kiss1r_gene]
    has_proxy_genes_in_var = all(g in adata.var_names for g in [gnaq_gene, *proxy_genes])
    if stratification_mode == "proxy":
        required_genes.extend([gnaq_gene, *proxy_genes])

    missing_genes = [g for g in required_genes if g not in adata.var_names]
    if missing_genes:
        raise KeyError(
            f"Missing required gene(s) in AnnData for '{stratification_mode}' mode: {missing_genes}"
        )

    # 1. Subject-level pseudobulk (groups_col=None aggregates all cells per subject)
    pdata = dc.pp.pseudobulk(
        adata,
        sample_col="ASAP_subject_id",
        groups_col=None,
        mode="sum",
    )

    n_samples_before_filter = int(pdata.n_obs)

    # 2. Keep only pseudobulk samples with at least min_cells
    dc.pp.filter_samples(pdata, min_cells=min_cells)
    n_samples_after_filter = int(pdata.n_obs)

    if n_samples_after_filter == 0:
        LOGGER.warning("No pseudobulk samples passed min_cells=%d for %s.", min_cells, cohort_label)
        return pdata, {
            "pseudobulk_samples_prefilter": n_samples_before_filter,
            "pseudobulk_samples_postfilter": 0,
            "high_count": 0,
            "low_count": 0,
        }

    # 3. Normalize a copy to evaluate gene expression and run decoupler
    pdata_norm = pdata.copy()
    sc.pp.normalize_total(pdata_norm, target_sum=target_sum)
    sc.pp.log1p(pdata_norm)

    kiss1r_expr = _extract_1d_expr(pdata_norm, kiss1r_gene)
    kiss1r_pos = kiss1r_expr > 0

    # Always store pure KISS1R stratification for cross-configuration stability comparison
    pdata_norm.obs["STRAT_kiss1r_pure"] = np.where(kiss1r_pos, "High", "Low")

    strat_stats: dict[str, Any] = {
        "stratification_mode": stratification_mode,
        "pseudobulk_samples_prefilter": n_samples_before_filter,
        "pseudobulk_samples_postfilter": n_samples_after_filter,
        "kiss1r_positive_samples": int(kiss1r_pos.sum()),
    }

    if has_proxy_genes_in_var:
        gnaq_expr = _extract_1d_expr(pdata_norm, gnaq_gene)
        gnaq_pos = gnaq_expr > 0

        proxy_pos_dict: dict[str, np.ndarray] = {
            pg: (_extract_1d_expr(pdata_norm, pg) > 0) for pg in proxy_genes
        }
        proxy_count = np.sum(
            [pos.astype(int) for pos in proxy_pos_dict.values()],
            axis=0,
        )

        # Store alternative proxy thresholds for configuration stability audit
        audit_thresholds = sorted(
            set([*config.get("stability_proxy_thresholds", []), min_proxies])
        )
        for k_thresh in audit_thresholds:
            if 1 <= k_thresh <= len(proxy_genes):
                is_high_k = kiss1r_pos | (gnaq_pos & (proxy_count >= k_thresh))
                pdata_norm.obs[f"STRAT_proxy_min{k_thresh}"] = np.where(
                    is_high_k, "High", "Low"
                )

    if stratification_mode == "proxy":
        is_high = kiss1r_pos | (gnaq_pos & (proxy_count >= min_proxies))

        strat_stats["gnaq_positive_samples"] = int(gnaq_pos.sum())
        strat_stats["min_active_proxies"] = min_proxies
        strat_stats["proxy_genes_positive_samples"] = {
            pg: int(pos.sum()) for pg, pos in proxy_pos_dict.items()
        }
        strat_stats["rescued_by_proxies_samples"] = int(
            ((~kiss1r_pos) & gnaq_pos & (proxy_count >= min_proxies)).sum()
        )
    elif stratification_mode == "kiss1r":
        is_high = kiss1r_pos
    else:
        raise ValueError(f"Invalid stratification_mode: {stratification_mode}")

    status_labels = np.where(is_high, "High", "Low")
    pdata.obs["PROXY_status"] = status_labels
    pdata_norm.obs["PROXY_status"] = status_labels

    strat_stats["high_count"] = int((status_labels == "High").sum())
    strat_stats["low_count"] = int((status_labels == "Low").sum())

    LOGGER.info(
        "%s stratification (%s) -> High: %d | Low: %d (out of %d samples)",
        cohort_label,
        stratification_mode,
        strat_stats["high_count"],
        strat_stats["low_count"],
        n_samples_after_filter,
    )

    return pdata_norm, strat_stats


def _evaluate_tf_matrix(
    tf_acts: pd.DataFrame,
    mask_high: np.ndarray,
    mask_low: np.ndarray,
) -> pd.DataFrame:
    """Computes mean difference (High - Low), Welch's t-test p-value, and BH-adjusted p-value."""
    high_acts = tf_acts.iloc[mask_high]
    low_acts = tf_acts.iloc[mask_low]

    diff = high_acts.mean(axis=0) - low_acts.mean(axis=0)
    _, p_vals = stats.ttest_ind(high_acts, low_acts, equal_var=False, axis=0)
    p_vals = np.nan_to_num(p_vals, nan=1.0)
    _, padj, _, _ = multipletests(p_vals, method="fdr_bh")

    return pd.DataFrame(
        {
            "delta_activity": diff.astype(float),
            "pval": pd.Series(p_vals, index=diff.index, dtype=float),
            "padj": pd.Series(padj, index=diff.index, dtype=float),
        },
        index=diff.index,
    )


def _extract_method_subset(
    res_all: dict[str, pd.DataFrame],
    methods_subset: list[str],
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Extracts score/padj matrices for the requested decoupler methods."""
    res_sub: dict[str, pd.DataFrame] = {}
    valid_methods: list[str] = []
    for method in methods_subset:
        score_key = f"score_{method}"
        padj_key = f"padj_{method}"
        if score_key in res_all:
            res_sub[score_key] = res_all[score_key]
            valid_methods.append(method)
        if padj_key in res_all:
            res_sub[padj_key] = res_all[padj_key]
    return res_sub, valid_methods


def evaluate_tf_consensus_and_stability(
    pdata_norm: ad.AnnData,
    cohort_label: str,
    net: pd.DataFrame,
    combinations: list[tuple[str, ...]],
    output_csv_base: Path,
    output_dir: Path,
    config: dict[str, Any],
    stratification_mode: str,
) -> dict[str, Any]:
    """
    1. Evaluates TF differential activity on a pre-specified primary consensus (avoiding post-selection bias).
    2. Exports the primary consensus TF measurements table for Step 05 (CARNIVAL).
    3. Audits TF stability across:
       - Individual decoupler methods
       - All method combinations (e.g., 219 combinations)
       - Alternative stratification configurations (pure KISS1R vs proxy thresholds)
    4. Exports a comprehensive stability table (`TF_Stability_*.csv`) for Step 06.
    """
    methods_all: list[str] = config["decoupler_methods"]
    primary_methods: list[str] = config["primary_consensus_methods"]
    run_stability_audit = bool(config.get("run_stability_audit", True))
    tmin_all = int(config["tmin_targets"])
    fdr_thresh = float(config["fdr_threshold"])
    nom_p_thresh = float(config.get("nominal_p_threshold", 0.05))
    diff_thresh = float(config["diff_threshold"])

    obs = pdata_norm.obs
    mask_high = (obs["PROXY_status"] == "High").to_numpy()
    mask_low = (obs["PROXY_status"] == "Low").to_numpy()

    n_high = int(mask_high.sum())
    n_low = int(mask_low.sum())

    if n_high < 2 or n_low < 2:
        LOGGER.warning(
            "Skipping Decoupler evaluation for %s: insufficient samples for Welch's t-test "
            "(High=%d, Low=%d; minimum required is 2 per group).",
            cohort_label,
            n_high,
            n_low,
        )
        return {
            "status": "skipped_insufficient_samples",
            "primary_combo": None,
            "primary_num_tfs": 0,
            "tested_combinations": 0,
            "best_combo": None,
            "best_num_tfs": 0,
            "output_csv": None,
            "stability_csv": None,
            "significant_tfs": {},
        }

    # Ensure all primary methods are included when running decouple
    methods_to_run = list(dict.fromkeys([*methods_all, *primary_methods]))
    LOGGER.info("Running dc.mt.decouple for %s with methods=%s...", cohort_label, methods_to_run)
    dc.mt.decouple(data=pdata_norm, net=net, methods=methods_to_run, tmin=tmin_all)

    res_all = {key: pdata_norm.obsm[key] for key in pdata_norm.obsm.keys()}

    # -------------------------------------------------------------------------
    # 1. PRIMARY PRE-SPECIFIED CONSENSUS INFERENCE (No cherry-picking)
    # -------------------------------------------------------------------------
    res_primary, valid_primary = _extract_method_subset(res_all, primary_methods)
    if not valid_primary:
        raise RuntimeError(
            f"None of the requested primary_consensus_methods {primary_methods} produced valid scores."
        )

    primary_combo_name = " + ".join(valid_primary)
    if len(valid_primary) == 1:
        primary_tf_acts = res_all[f"score_{valid_primary[0]}"]
    else:
        primary_tf_acts, _ = dc.mt.consensus(res_primary)

    primary_stats_df = _evaluate_tf_matrix(primary_tf_acts, mask_high, mask_low)
    primary_sig_mask = (primary_stats_df["padj"] < fdr_thresh) & (
        primary_stats_df["delta_activity"].abs() > diff_thresh
    )
    primary_nom_mask = (primary_stats_df["pval"] < nom_p_thresh) & (
        primary_stats_df["delta_activity"].abs() > diff_thresh
    )

    primary_valid_df = primary_stats_df.loc[primary_sig_mask].sort_values(
        by="delta_activity", ascending=False
    )
    primary_num_tfs = int(len(primary_valid_df))
    primary_max_abs_diff = (
        float(primary_stats_df["delta_activity"].abs().max())
        if not primary_stats_df.empty
        else 0.0
    )

    out_csv_path = build_tagged_path(
        output_csv_base,
        cohort_label,
        stratification_mode.upper(),
        default_dir=output_dir,
    )
    stability_csv_path = build_tagged_path(
        Path("TF_Stability.csv"),
        cohort_label,
        stratification_mode.upper(),
        default_dir=output_dir,
    )

    if primary_num_tfs > 0:
        df_carnival_primary = pd.DataFrame(
            {
                "Node": primary_valid_df.index,
                "Measurement": primary_valid_df["delta_activity"].values,
            }
        )
        df_carnival_primary.to_csv(out_csv_path, index=False)
        LOGGER.info(
            "[%s] Primary pre-specified consensus (%s): %d significant TFs (FDR < %.2f, |diff| > %.2f) -> %s",
            cohort_label,
            primary_combo_name,
            primary_num_tfs,
            fdr_thresh,
            diff_thresh,
            out_csv_path,
        )
    else:
        LOGGER.warning(
            "[%s] Primary pre-specified consensus (%s) yielded 0 TFs at FDR < %.2f and |diff| > %.2f.",
            cohort_label,
            primary_combo_name,
            fdr_thresh,
            diff_thresh,
        )

    # -------------------------------------------------------------------------
    # 2. SINGLE-METHOD STABILITY AUDIT
    # -------------------------------------------------------------------------
    single_method_stats: dict[str, pd.DataFrame] = {}
    for method in methods_to_run:
        score_key = f"score_{method}"
        if score_key in res_all:
            try:
                single_method_stats[method] = _evaluate_tf_matrix(
                    res_all[score_key], mask_high, mask_low
                )
            except Exception as exc:
                LOGGER.warning(
                    "Could not compute single-method stats for %s in %s: %s",
                    method,
                    cohort_label,
                    exc,
                )

    # -------------------------------------------------------------------------
    # 3. COMBINATORIAL STABILITY AUDIT (Across all 219 method combinations)
    # -------------------------------------------------------------------------
    all_tfs = primary_stats_df.index
    combo_fdr_pass_count = pd.Series(0, index=all_tfs, dtype=int)
    combo_nom_pass_count = pd.Series(0, index=all_tfs, dtype=int)
    combo_padj_records: dict[str, list[float]] = {str(tf): [] for tf in all_tfs}
    combo_delta_records: dict[str, list[float]] = {str(tf): [] for tf in all_tfs}

    best_num_tfs = 0
    best_max_abs_diff = 0.0
    best_combo: str | None = None
    tested_combos = 0

    if run_stability_audit:
        LOGGER.info(
            "Running stability audit across %d method combinations for %s...",
            len(combinations),
            cohort_label,
        )
        for combo in combinations:
            res_combo, valid_methods = _extract_method_subset(res_all, list(combo))
            if len(valid_methods) < 2:
                continue

            combo_name = " + ".join(valid_methods)
            try:
                tf_acts_c, _ = dc.mt.consensus(res_combo)
                c_stats = _evaluate_tf_matrix(tf_acts_c, mask_high, mask_low)

                fdr_mask = (c_stats["padj"] < fdr_thresh) & (
                    c_stats["delta_activity"].abs() > diff_thresh
                )
                nom_mask = (c_stats["pval"] < nom_p_thresh) & (
                    c_stats["delta_activity"].abs() > diff_thresh
                )

                passed_fdr_tfs = c_stats.index[fdr_mask]
                passed_nom_tfs = c_stats.index[nom_mask]

                combo_fdr_pass_count.loc[passed_fdr_tfs] += 1
                combo_nom_pass_count.loc[passed_nom_tfs] += 1

                for tf in c_stats.index:
                    tf_str = str(tf)
                    if tf_str in combo_padj_records:
                        combo_padj_records[tf_str].append(float(c_stats.loc[tf, "padj"]))
                        combo_delta_records[tf_str].append(
                            float(c_stats.loc[tf, "delta_activity"])
                        )

                tested_combos += 1
                num_sig = int(fdr_mask.sum())
                current_max_diff = (
                    float(c_stats["delta_activity"].abs().max()) if not c_stats.empty else 0.0
                )

                if num_sig > best_num_tfs or (
                    num_sig > 0
                    and num_sig == best_num_tfs
                    and current_max_diff > best_max_abs_diff
                ):
                    best_num_tfs = num_sig
                    best_max_abs_diff = current_max_diff
                    best_combo = combo_name

            except Exception as exc:
                LOGGER.warning(
                    "Error evaluating combination [%s] in %s: %s: %s",
                    combo_name,
                    cohort_label,
                    type(exc).__name__,
                    exc,
                )

    # -------------------------------------------------------------------------
    # 4. CROSS-CONFIGURATION STABILITY AUDIT (Pure KISS1R & Proxy Thresholds)
    # -------------------------------------------------------------------------
    alt_config_stats: dict[str, pd.DataFrame] = {}
    strat_cols = [col for col in obs.columns if col.startswith("STRAT_")]
    for scol in strat_cols:
        alt_high = (obs[scol] == "High").to_numpy()
        alt_low = (obs[scol] == "Low").to_numpy()
        if int(alt_high.sum()) >= 2 and int(alt_low.sum()) >= 2:
            cfg_name = scol.replace("STRAT_", "")
            alt_config_stats[cfg_name] = _evaluate_tf_matrix(
                primary_tf_acts, alt_high, alt_low
            )

    # -------------------------------------------------------------------------
    # 5. BUILD COMPREHENSIVE STABILITY TABLE FOR STEP 06
    # -------------------------------------------------------------------------
    n_combos_denom = max(tested_combos, 1)
    n_configs_denom = max(len(alt_config_stats), 1)
    stability_rows: list[dict[str, Any]] = []

    for tf in all_tfs:
        tf_str = str(tf)
        p_row = primary_stats_df.loc[tf]
        primary_delta = float(p_row["delta_activity"])
        primary_sign = np.sign(primary_delta)

        # Single-method agreement metrics
        n_single_valid = 0
        n_same_sign = 0
        nom_methods: list[str] = []
        fdr_methods: list[str] = []
        per_method_cols: dict[str, float] = {}

        for m, m_df in single_method_stats.items():
            if tf not in m_df.index:
                continue
            n_single_valid += 1
            m_delta = float(m_df.loc[tf, "delta_activity"])
            m_pval = float(m_df.loc[tf, "pval"])
            m_padj = float(m_df.loc[tf, "padj"])

            per_method_cols[f"delta_{m}"] = round(m_delta, 6)
            per_method_cols[f"pval_{m}"] = round(m_pval, 6)
            per_method_cols[f"padj_{m}"] = round(m_padj, 6)

            if np.sign(m_delta) == primary_sign and primary_sign != 0:
                n_same_sign += 1
                if m_pval < nom_p_thresh and abs(m_delta) > diff_thresh:
                    nom_methods.append(m)
                if m_padj < fdr_thresh and abs(m_delta) > diff_thresh:
                    fdr_methods.append(m)

        # Combinatorial stability metrics
        padj_list = combo_padj_records.get(tf_str, [])
        delta_list = combo_delta_records.get(tf_str, [])
        median_padj = float(np.median(padj_list)) if padj_list else float(p_row["padj"])
        iqr_padj = (
            float(np.percentile(padj_list, 75) - np.percentile(padj_list, 25))
            if padj_list
            else 0.0
        )
        min_padj = float(np.min(padj_list)) if padj_list else float(p_row["padj"])
        median_delta = float(np.median(delta_list)) if delta_list else primary_delta

        # Cross-configuration stability metrics
        n_cfg_same_sign = 0
        n_cfg_nom_sig = 0
        n_cfg_fdr_sig = 0
        kiss1r_pure_delta = np.nan
        kiss1r_pure_pval = np.nan
        kiss1r_pure_padj = np.nan

        for cfg_name, cfg_df in alt_config_stats.items():
            if tf not in cfg_df.index:
                continue
            c_delta = float(cfg_df.loc[tf, "delta_activity"])
            c_pval = float(cfg_df.loc[tf, "pval"])
            c_padj = float(cfg_df.loc[tf, "padj"])

            if cfg_name == "kiss1r_pure":
                kiss1r_pure_delta = round(c_delta, 6)
                kiss1r_pure_pval = round(c_pval, 6)
                kiss1r_pure_padj = round(c_padj, 6)

            if np.sign(c_delta) == primary_sign and primary_sign != 0:
                n_cfg_same_sign += 1
                if c_pval < nom_p_thresh and abs(c_delta) > diff_thresh:
                    n_cfg_nom_sig += 1
                if c_padj < fdr_thresh and abs(c_delta) > diff_thresh:
                    n_cfg_fdr_sig += 1

        row_dict: dict[str, Any] = {
            "Node": tf_str,
            "primary_delta_activity": round(primary_delta, 6),
            "primary_pval": round(float(p_row["pval"]), 6),
            "primary_padj": round(float(p_row["padj"]), 6),
            "primary_significant": bool(primary_sig_mask.loc[tf]),
            "primary_nominal_significant": bool(primary_nom_mask.loc[tf]),
            "n_single_methods_tested": n_single_valid,
            "n_single_methods_same_sign": n_same_sign,
            "single_method_sign_concordance": round(
                n_same_sign / max(n_single_valid, 1), 4
            ),
            "n_single_methods_nominal_sig": len(nom_methods),
            "single_methods_nominal_sig": ";".join(nom_methods),
            "n_single_methods_fdr_sig": len(fdr_methods),
            "single_methods_fdr_sig": ";".join(fdr_methods),
            "combo_fdr_pass_count": int(combo_fdr_pass_count.loc[tf]),
            "combo_fdr_selection_freq": round(
                float(combo_fdr_pass_count.loc[tf]) / n_combos_denom, 4
            ),
            "combo_nominal_pass_count": int(combo_nom_pass_count.loc[tf]),
            "combo_nominal_selection_freq": round(
                float(combo_nom_pass_count.loc[tf]) / n_combos_denom, 4
            ),
            "combo_median_delta": round(median_delta, 6),
            "combo_median_padj": round(median_padj, 6),
            "combo_iqr_padj": round(iqr_padj, 6),
            "combo_min_padj": round(min_padj, 6),
            "n_configs_tested": len(alt_config_stats),
            "config_sign_concordance_freq": round(
                n_cfg_same_sign / n_configs_denom, 4
            ),
            "config_nominal_stability_freq": round(
                n_cfg_nom_sig / n_configs_denom, 4
            ),
            "config_fdr_stability_freq": round(
                n_cfg_fdr_sig / n_configs_denom, 4
            ),
            "kiss1r_pure_delta": kiss1r_pure_delta,
            "kiss1r_pure_pval": kiss1r_pure_pval,
            "kiss1r_pure_padj": kiss1r_pure_padj,
        }
        row_dict.update(per_method_cols)
        stability_rows.append(row_dict)

    stability_df = pd.DataFrame(stability_rows).sort_values(
        by=["primary_significant", "primary_padj", "combo_fdr_selection_freq"],
        ascending=[False, True, False],
    )
    stability_df.to_csv(stability_csv_path, index=False)
    LOGGER.info(
        "[%s] Saved TF stability audit table (%d TFs) -> %s",
        cohort_label,
        len(stability_df),
        stability_csv_path,
    )

    # Detailed dict for significant TFs in the primary consensus
    stability_indexed = stability_df.set_index("Node")
    significant_tfs_detail: dict[str, dict[str, Any]] = {}
    for tf in primary_valid_df.index:
        tf_str = str(tf)
        s_row = stability_indexed.loc[tf_str]
        significant_tfs_detail[tf_str] = {
            "delta_activity": float(s_row["primary_delta_activity"]),
            "pval": float(s_row["primary_pval"]),
            "padj": float(s_row["primary_padj"]),
            "single_method_sign_concordance": float(
                s_row["single_method_sign_concordance"]
            ),
            "single_methods_nominal_sig": str(s_row["single_methods_nominal_sig"]),
            "combo_fdr_selection_freq": float(s_row["combo_fdr_selection_freq"]),
            "combo_nominal_selection_freq": float(s_row["combo_nominal_selection_freq"]),
            "combo_median_padj": float(s_row["combo_median_padj"]),
            "config_sign_concordance_freq": float(s_row["config_sign_concordance_freq"]),
            "config_nominal_stability_freq": float(s_row["config_nominal_stability_freq"]),
            "kiss1r_pure_delta": (
                None
                if pd.isna(s_row["kiss1r_pure_delta"])
                else float(s_row["kiss1r_pure_delta"])
            ),
            "kiss1r_pure_pval": (
                None
                if pd.isna(s_row["kiss1r_pure_pval"])
                else float(s_row["kiss1r_pure_pval"])
            ),
        }

    # Also summarize top stable TFs across combinations/methods even if they missed strict primary FDR
    top_stable_candidates = (
        stability_df.sort_values(
            by=["combo_fdr_selection_freq", "combo_nominal_selection_freq", "primary_pval"],
            ascending=[False, False, True],
        )
        .head(15)["Node"]
        .tolist()
    )

    return {
        "status": "completed",
        "primary_combo": primary_combo_name,
        "primary_num_tfs": primary_num_tfs,
        "primary_max_abs_diff": round(primary_max_abs_diff, 6),
        "tested_combinations": tested_combos,
        "exploratory_best_combo": best_combo,
        "exploratory_best_num_tfs": best_num_tfs,
        "exploratory_best_max_abs_diff": round(best_max_abs_diff, 6),
        # Backward-compatible keys pointing to the pre-specified primary consensus
        "best_combo": primary_combo_name,
        "best_num_tfs": primary_num_tfs,
        "best_max_abs_diff": round(primary_max_abs_diff, 6),
        "output_csv": str(out_csv_path) if primary_num_tfs > 0 else None,
        "stability_csv": str(stability_csv_path),
        "significant_tfs": significant_tfs_detail,
        "top_stable_tfs_across_audit": top_stable_candidates,
    }


def format_pipeline_step04_summary(
    *,
    dataset_path: str,
    config_path: str,
    cohorts: list[str],
    stratification_mode: str,
    primary_methods: list[str],
    output_dir: str,
    report_path: str,
) -> None:
    LOGGER.info(
        "Pipeline Step04 summary:\n"
        f"- Dopaminergic Dataset: {dataset_path}\n"
        f"- Config File: {config_path}\n"
        f"- Analyzed Cohorts: {', '.join(cohorts)}\n"
        f"- Stratification Mode: {stratification_mode}\n"
        f"- Pre-specified Primary Consensus: {' + '.join(primary_methods)}\n"
        f"- Output Directory: {output_dir}\n"
        f"- Report Path: {report_path}"
    )


def build_step04_parser(default_repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step04 Pseudobulk KISS1R/Proxy Stratification & Pre-specified Decoupler Consensus with Stability Audit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    processed_dir = default_repo_root / "pipeline" / "data" / "processed"
    reports_dir = default_repo_root / "pipeline" / "reports"

    parser.add_argument(
        "--config",
        default=str(default_repo_root / "pipeline" / "config" / "step04.yml"),
        help="YAML config file for step04 tunables.",
    )
    parser.add_argument(
        "--dataset-path",
        default=str(processed_dir / CANONICAL_NAMES["dataset_dopaminergic"]),
        help="Path to the dopaminergic dataset file (.h5ad).",
    )
    parser.add_argument(
        "--cohorts",
        nargs="+",
        choices=VALID_COHORTS,
        default=["pd_male"],
        help=(
            "One or more target subgroups to analyze: 'control_male', 'control_female', "
            "'pd_male', 'pd_female', 'male', 'female', or 'all' to run all six subgroups."
        ),
    )
    parser.add_argument(
        "--stratification",
        choices=["proxy", "kiss1r"],
        default=None,
        help=(
            "Stratification strategy for High/Low groups: 'proxy' (KISS1R or GNAQ + proxies) "
            "or 'kiss1r' (KISS1R expression > 0 only). Overrides config if provided."
        ),
    )
    parser.add_argument(
        "--min-cells",
        type=int,
        default=None,
        help="Minimum number of cells per subject in pseudobulk (overrides config).",
    )
    parser.add_argument(
        "--carnival-csv-name",
        default=CANONICAL_NAMES["transcription_factors_name"],
        help="Base filename for the exported CARNIVAL TF measurements CSV.",
    )
    parser.add_argument("--output-dir", default=str(processed_dir / "step04_carnival"))
    parser.add_argument("--reports-dir", default=str(reports_dir))
    parser.add_argument("--logging-level", default="INFO")
    return parser


def run_step04(args: argparse.Namespace) -> None:
    configure_logging(args.logging_level)
    config_path = Path(args.config)
    config = _resolve_step04_config(config_path)

    min_cells = (
        args.min_cells if args.min_cells is not None else int(config["min_cells_pseudobulk"])
    )
    stratification_mode = (
        args.stratification
        if args.stratification is not None
        else str(config["stratification_mode"]).lower()
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    adata = sc.read_h5ad(dataset_path)

    # Expand 'all' if requested
    requested_cohorts: list[str] = []
    for c in args.cohorts:
        if c == "all":
            requested_cohorts.extend(
                ["control_male", "control_female", "pd_male", "pd_female", "male", "female"]
            )
        else:
            requested_cohorts.append(c)
    requested_cohorts = list(dict.fromkeys(requested_cohorts))

    # Pre-load DoRothEA network and pre-compute method combinations once
    dorothea_levels: list[str] = config["dorothea_levels"]
    LOGGER.info("Loading DoRothEA network (levels=%s)...", dorothea_levels)
    net = dc.op.dorothea(levels=dorothea_levels)

    methods_all: list[str] = config["decoupler_methods"]
    primary_methods: list[str] = config["primary_consensus_methods"]
    min_combo_size = int(config["min_combo_size"])
    combinations: list[tuple[str, ...]] = []
    for r in range(min_combo_size, len(methods_all) + 1):
        combinations.extend(itertools.combinations(methods_all, r))

    report: dict[str, Any] = {
        "config_path": str(config_path),
        "dataset_path": str(dataset_path),
        "stratification_mode": stratification_mode,
        "parameters": {
            "min_cells_pseudobulk": min_cells,
            "target_sum_norm": float(config["target_sum_norm"]),
            "kiss1r_gene": config["kiss1r_gene"],
            "gnaq_gene": config["gnaq_gene"] if stratification_mode == "proxy" else None,
            "proxy_genes": config["proxy_genes"] if stratification_mode == "proxy" else [],
            "min_active_proxies": (
                int(config["min_active_proxies"]) if stratification_mode == "proxy" else None
            ),
            "stability_proxy_thresholds": config["stability_proxy_thresholds"],
            "dorothea_levels": dorothea_levels,
            "primary_consensus_methods": primary_methods,
            "decoupler_methods": methods_all,
            "run_stability_audit": bool(config["run_stability_audit"]),
            "min_combo_size": min_combo_size,
            "total_combinations_audited": len(combinations),
            "tmin_targets": int(config["tmin_targets"]),
            "fdr_threshold": float(config["fdr_threshold"]),
            "nominal_p_threshold": float(config["nominal_p_threshold"]),
            "diff_threshold": float(config["diff_threshold"]),
        },
        "cohorts": {},
    }

    for cohort_key in requested_cohorts:
        cohort_label, sub_adata = _subset_adata_by_cohort(adata, cohort_key)
        LOGGER.info("=== Processing Cohort: %s (cells=%d) ===", cohort_label, sub_adata.n_obs)

        if sub_adata.n_obs == 0:
            LOGGER.warning("No cells found for cohort %s. Skipping...", cohort_label)
            report["cohorts"][cohort_label] = {"status": "empty_cohort", "total_cells": 0}
            continue

        pdata_norm, strat_stats = make_pseudobulk_stratified(
            adata=sub_adata,
            cohort_label=cohort_label,
            config=config,
            min_cells=min_cells,
            stratification_mode=stratification_mode,
        )

        consensus_stats = evaluate_tf_consensus_and_stability(
            pdata_norm=pdata_norm,
            cohort_label=cohort_label,
            net=net,
            combinations=combinations,
            output_csv_base=Path(args.carnival_csv_name),
            output_dir=out_dir,
            config=config,
            stratification_mode=stratification_mode,
        )

        report["cohorts"][cohort_label] = {
            "cohort_key": cohort_key,
            "total_cells": int(sub_adata.n_obs),
            "total_subjects_raw": int(sub_adata.obs["ASAP_subject_id"].nunique()),
            "stratification": strat_stats,
            "decoupler_consensus": consensus_stats,
        }

    report_path = reports_dir / "step04_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    format_pipeline_step04_summary(
        dataset_path=str(dataset_path),
        config_path=str(config_path),
        cohorts=requested_cohorts,
        stratification_mode=stratification_mode,
        primary_methods=primary_methods,
        output_dir=str(out_dir),
        report_path=str(report_path),
    )