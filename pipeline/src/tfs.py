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
from pipeline.src.utils import build_tagged_path, configure_logging, load_yaml_file, _extract_1d_expr, _ensure_list

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
    "proxy_genes": ["CALB2", "TH", "S100B", "GNRH1"],
    "min_active_proxies": 3,
    "dorothea_levels": ["A", "B", "C"],
    "decoupler_methods": ["ulm", "waggr", "viper", "aucell", "gsea", "gsva", "mlm", "zscore"],
    "min_combo_size": 3,
    "tmin_targets": 6,
    "fdr_threshold": 0.05,
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

    strat_stats: dict[str, Any] = {
        "stratification_mode": stratification_mode,
        "pseudobulk_samples_prefilter": n_samples_before_filter,
        "pseudobulk_samples_postfilter": n_samples_after_filter,
        "kiss1r_positive_samples": int(kiss1r_pos.sum()),
    }

    if stratification_mode == "proxy":
        gnaq_expr = _extract_1d_expr(pdata_norm, gnaq_gene)
        gnaq_pos = gnaq_expr > 0

        proxy_pos_dict: dict[str, np.ndarray] = {}
        for pg in proxy_genes:
            proxy_pos_dict[pg] = _extract_1d_expr(pdata_norm, pg) > 0

        proxy_count = np.sum(
            [pos.astype(int) for pos in proxy_pos_dict.values()],
            axis=0,
        )
        is_high = kiss1r_pos | (gnaq_pos & (proxy_count >= min_proxies))

        strat_stats["gnaq_positive_samples"] = int(gnaq_pos.sum())
        strat_stats["min_active_proxies"] = min_proxies
        strat_stats["proxy_genes_positive_samples"] = {
            pg: int(pos.sum()) for pg, pos in proxy_pos_dict.items()
        }
        strat_stats[" rescued_by_proxies_samples"] = int(
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


def optimize_tf_consensus(
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
    Runs decoupler methods on normalized pseudobulk profiles, searches over all method
    combinations for the consensus maximizing significant differential TFs (High vs Low),
    and exports the best CARNIVAL measurement table.
    """
    methods_all: list[str] = config["decoupler_methods"]
    tmin_all = int(config["tmin_targets"])
    fdr_thresh = float(config["fdr_threshold"])
    diff_thresh = float(config["diff_threshold"])

    obs = pdata_norm.obs
    mask_high = (obs["PROXY_status"] == "High").to_numpy()
    mask_low = (obs["PROXY_status"] == "Low").to_numpy()

    n_high = int(mask_high.sum())
    n_low = int(mask_low.sum())

    if n_high < 2 or n_low < 2:
        LOGGER.warning(
            "Skipping Decoupler optimization for %s: insufficient samples for Welch's t-test "
            "(High=%d, Low=%d; minimum required is 2 per group).",
            cohort_label,
            n_high,
            n_low,
        )
        return {
            "status": "skipped_insufficient_samples",
            "tested_combinations": 0,
            "best_combo": None,
            "best_num_tfs": 0,
            "output_csv": None,
            "significant_tfs": {},
        }

    LOGGER.info("Running dc.mt.decouple for %s with methods=%s...", cohort_label, methods_all)
    dc.mt.decouple(data=pdata_norm, net=net, methods=methods_all, tmin=tmin_all)

    res_all = {key: pdata_norm.obsm[key] for key in pdata_norm.obsm.keys()}

    best_num_tfs = 0
    best_max_abs_diff = 0.0
    best_combo: str | None = None
    best_df_carnival: pd.DataFrame | None = None
    best_tfs_detail: dict[str, dict[str, float]] = {}
    tested_combos = 0

    out_csv_path = build_tagged_path(
        output_csv_base,
        cohort_label,
        stratification_mode.upper(),
        default_dir=output_dir,
    )

    for combo in combinations:
        methods_combo = list(combo)
        combo_name = " + ".join(methods_combo)

        res_combo: dict[str, Any] = {}
        valid_methods = 0
        for method in methods_combo:
            score_key = f"score_{method}"
            padj_key = f"padj_{method}"
            if score_key in res_all:
                res_combo[score_key] = res_all[score_key]
                valid_methods += 1
            if padj_key in res_all:
                res_combo[padj_key] = res_all[padj_key]

        if valid_methods < 2:
            continue

        try:
            tf_acts, _ = dc.mt.consensus(res_combo)

            high_acts = tf_acts.iloc[mask_high]
            low_acts = tf_acts.iloc[mask_low]

            # Mean activity difference (High - Low)
            diff = high_acts.mean(axis=0) - low_acts.mean(axis=0)

            # Welch's t-test across samples for each TF
            _, p_vals = stats.ttest_ind(high_acts, low_acts, equal_var=False, axis=0)
            p_vals = np.nan_to_num(p_vals, nan=1.0)

            # Benjamini-Hochberg FDR correction
            _, padj, _, _ = multipletests(p_vals, method="fdr_bh")
            pvals_sig = pd.Series(padj, index=diff.index)

            mask_sig = (pvals_sig < fdr_thresh) & (diff.abs() > diff_thresh)
            valid_tfs = diff[mask_sig].sort_values(ascending=False)

            tested_combos += 1
            num_sig = int(len(valid_tfs))
            current_max_diff = float(diff.abs().max()) if not diff.empty else 0.0

            # Update record if strictly more TFs found (or tie broken by larger max |diff|)
            if num_sig > best_num_tfs or (
                num_sig > 0 and num_sig == best_num_tfs and current_max_diff > best_max_abs_diff
            ):
                best_num_tfs = num_sig
                best_max_abs_diff = current_max_diff
                best_combo = combo_name

                best_df_carnival = pd.DataFrame(
                    {
                        "Node": valid_tfs.index,
                        "Measurement": valid_tfs.values,
                        "padj": pvals_sig.loc[valid_tfs.index].values,
                    }
                )

                # Save 2-column CARNIVAL input CSV immediately
                best_df_carnival[["Node", "Measurement"]].to_csv(out_csv_path, index=False)

                best_tfs_detail = {
                    str(tf): {
                        "delta_activity": round(float( valid_tfs.loc[tf]), 6),
                        "padj": round(float(pvals_sig.loc[tf]), 6),
                    }
                    for tf in valid_tfs.index
                }

                LOGGER.info(
                    "[%s] New optimal consensus: %d TFs (Combo: %s, max |diff|=%.3f) -> %s",
                    cohort_label,
                    num_sig,
                    combo_name,
                    current_max_diff,
                    out_csv_path,
                )

        except Exception as exc:
            LOGGER.warning(
                "Error evaluating combination [%s] in %s: %s: %s",
                combo_name,
                cohort_label,
                type(exc).__name__,
                exc,
            )

    return {
        "status": "completed",
        "tested_combinations": tested_combos,
        "best_combo": best_combo,
        "best_num_tfs": best_num_tfs,
        "best_max_abs_diff": round(best_max_abs_diff, 6),
        "output_csv": str(out_csv_path) if best_num_tfs > 0 else None,
        "significant_tfs": best_tfs_detail,
    }


def format_pipeline_step04_summary(
    *,
    dataset_path: str,
    config_path: str,
    cohorts: list[str],
    stratification_mode: str,
    output_dir: str,
    report_path: str,
) -> None:
    LOGGER.info(
        "Pipeline Step04 summary:\n"
        f"- Dopaminergic Dataset: {dataset_path}\n"
        f"- Config File: {config_path}\n"
        f"- Analyzed Cohorts: {', '.join(cohorts)}\n"
        f"- Stratification Mode: {stratification_mode}\n"
        f"- Output Directory: {output_dir}\n"
        f"- Report Path: {report_path}"
    )


def build_step04_parser(default_repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step04 Pseudobulk KISS1R/Proxy Stratification & Decoupler Consensus TF Activity",
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
    # Deduplicate preserving order
    requested_cohorts = list(dict.fromkeys(requested_cohorts))

    # Pre-load DoRothEA network and pre-compute method combinations once
    dorothea_levels: list[str] = config["dorothea_levels"]
    LOGGER.info("Loading DoRothEA network (levels=%s)...", dorothea_levels)
    net = dc.op.dorothea(levels=dorothea_levels)

    methods_all: list[str] = config["decoupler_methods"]
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
            "dorothea_levels": dorothea_levels,
            "decoupler_methods": methods_all,
            "min_combo_size": min_combo_size,
            "total_combinations_generated": len(combinations),
            "tmin_targets": int(config["tmin_targets"]),
            "fdr_threshold": float(config["fdr_threshold"]),
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

        opt_stats = optimize_tf_consensus(
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
            "decoupler_consensus": opt_stats,
        }

    report_path = reports_dir / "step04_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    format_pipeline_step04_summary(
        dataset_path=str(dataset_path),
        config_path=str(config_path),
        cohorts=requested_cohorts,
        stratification_mode=stratification_mode,
        output_dir=str(out_dir),
        report_path=str(report_path),
    )