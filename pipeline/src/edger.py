from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import anndata as ad
import decoupler as dc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pertpy as pt
import scanpy as sc
import rpy2.robjects as ro

from pipeline.src.schemas import CANONICAL_NAMES
from pipeline.src.utils import build_tagged_path, configure_logging, load_yaml_file

LOGGER = logging.getLogger(__name__)

DEFAULT_STEP03_CONFIG: dict[str, Any] = {
    "dpi_figures": 300,
    "min_cells_pseudobulk": 20,
    "min_counts_pseudobulk": 1000,
    "min_subjects_group": 5,
    "fdr_threshold": 0.05,
    "logfc_threshold": 0.3,
    "most_significant_dea": 10,
}


def _resolve_step03_config(config_path: Path) -> dict[str, Any]:
    config = DEFAULT_STEP03_CONFIG.copy()
    if config_path.exists():
        loaded = load_yaml_file(config_path) or {}
        step03_cfg = loaded.get("step03", {})
        if step03_cfg and not isinstance(step03_cfg, dict):
            raise ValueError(f"Expected mapping under 'step03' in config: {config_path}")
        config.update({k: v for k, v in step03_cfg.items() if v is not None})
    else:
        LOGGER.warning("Config file not found (%s), using default values.", config_path)
    return config


def _sanitize_name(name: str) -> str:
    """Sanitizes cell type names for safe file system paths."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")


def _detect_edger_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    """Dynamically detects logFC, p-value, and adjusted p-value columns from pertpy/EdgeR output."""
    lfc_col = next((c for c in ["log_fc", "logFC", "log2FoldChange"] if c in df.columns), None)
    pval_col = next((c for c in ["p_value", "PValue", "pvalue"] if c in df.columns), None)
    padj_col = next((c for c in ["adj_p_value", "FDR", "padj"] if c in df.columns), None)

    if not lfc_col or not pval_col:
        raise KeyError(
            f"Unable to locate logFC or p-value columns in EdgeR output: {list(df.columns)}"
        )
    return lfc_col, pval_col, padj_col or pval_col


def prepare_gsea_rnk(edger_results: pd.DataFrame, output_rnk_path: Path) -> None:
    """Generates and saves a .rnk file for WebGestalt/GSEA ranked by sign(logFC) * -log10(pval)."""
    lfc_col, pval_col, _ = _detect_edger_columns(edger_results)

    df = edger_results[[lfc_col, pval_col]].copy()
    df[lfc_col] = pd.to_numeric(df[lfc_col], errors="coerce")
    df[pval_col] = pd.to_numeric(df[pval_col], errors="coerce")
    df = df.dropna(subset=[lfc_col, pval_col])

    # Avoid log10(0) by clipping p-values to the smallest positive representable float
    pvals_clipped = np.clip(df[pval_col].to_numpy(), np.finfo(float).tiny, 1.0)
    df["score"] = np.sign(df[lfc_col].to_numpy()) * (-np.log10(pvals_clipped))

    df = df.sort_values("score", ascending=False)
    df[["score"]].to_csv(output_rnk_path, sep="\t", header=False)


def make_pseudobulk(
    adata: ad.AnnData,
    cohort_label: str,
    min_cells: int = 20,
    min_counts: int = 1000,
    qc_plot_path: Path | None = None,
    dpi: int = 300,
) -> ad.AnnData:
    """Generates pseudobulk profiles, saves QC plot before filtering, and filters low-quality samples."""
    LOGGER.info("Generating pseudobulk for %s (min_cells=%d)...", cohort_label, min_cells)

    # Se i conteggi grezzi sono in un layer, aggiungi layer="counts"
    pdata = dc.pp.pseudobulk(
        adata,
        sample_col="ASAP_subject_id",
        groups_col="HMBA_Group_name",
        mode="sum",
    )

    # Plot QC PRIMA di rimuovere i campioni filtrati
    if qc_plot_path is not None:
        try:
            dc.pl.filter_samples(
                pdata,
                groupby=["condition_id", "HMBA_Group_name"],
                min_cells=min_cells,
                min_counts=min_counts,
                dpi=dpi,
                save=str(qc_plot_path),
            )
            plt.close("all")
        except Exception as exc:
            LOGGER.warning("Skipped decoupler QC plot for %s: %s", cohort_label, exc)

    dc.pp.filter_samples(pdata, min_cells=min_cells, min_counts=min_counts)
    return pdata


def save_plot_volcano(
    df_res: pd.DataFrame,
    cohort_label: str,
    cell_type: str,
    output_path: Path,
    config: dict[str, Any],
) -> None:
    """Saves a Volcano plot for the EdgeR differential expression results of a specific cell type."""
    lfc_col, _, padj_col = _detect_edger_columns(df_res)
    fdr_thresh = float(config["fdr_threshold"])
    lfc_thresh = float(config["logfc_threshold"])
    dpi = int(config["dpi_figures"])

    df = df_res[[lfc_col, padj_col]].dropna().copy()
    if df.empty:
        return

    lfc = df[lfc_col].to_numpy(dtype=float)
    padj = np.clip(df[padj_col].to_numpy(dtype=float), np.finfo(float).tiny, 1.0)
    neg_log10_padj = -np.log10(padj)

    up_mask = (padj < fdr_thresh) & (lfc >= lfc_thresh)
    down_mask = (padj < fdr_thresh) & (lfc <= -lfc_thresh)
    ns_mask = ~(up_mask | down_mask)

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    ax.scatter(
        lfc[ns_mask],
        neg_log10_padj[ns_mask],
        c="lightgrey",
        alpha=0.6,
        s=15,
        label="NS",
    )
    ax.scatter(
        lfc[down_mask],
        neg_log10_padj[down_mask],
        c="#1f77b4",
        alpha=0.8,
        s=22,
        label=f"Down ({down_mask.sum()})",
    )
    ax.scatter(
        lfc[up_mask],
        neg_log10_padj[up_mask],
        c="#d62728",
        alpha=0.8,
        s=22,
        label=f"Up ({up_mask.sum()})",
    )

    ax.axhline(-np.log10(fdr_thresh), color="black", linestyle="--", linewidth=0.8)
    ax.axvline(lfc_thresh, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(-lfc_thresh, color="black", linestyle="--", linewidth=0.8)

    # Annotate top most significant DE genes
    sig_df = df[up_mask | down_mask].sort_values(padj_col, ascending=True).head(config["most_significant_dea"])
    for gene_name, row in sig_df.iterrows():
        y_val = -np.log10(max(float(row[padj_col]), np.finfo(float).tiny))
        ax.annotate(
            str(gene_name),
            (float(row[lfc_col]), y_val),
            fontsize=8,
            xytext=(3, 3),
            textcoords="offset points",
        )

    ax.set_title(f"Volcano Plot: {cell_type} ({cohort_label} - PD vs Control)")
    ax.set_xlabel("log2 Fold Change")
    ax.set_ylabel("-log10(FDR)")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_plot_deg_summary(
    results_by_celltype: dict[str, pd.DataFrame],
    cohort_label: str,
    output_path: Path,
    config: dict[str, Any],
) -> None:
    """Saves a summary barplot showing the count of Up- and Down-regulated DE genes per cell type."""
    if not results_by_celltype:
        return

    fdr_thresh = float(config["fdr_threshold"])
    lfc_thresh = float(config["logfc_threshold"])
    dpi = int(config["dpi_figures"])

    rows = []
    for cell_type, df in results_by_celltype.items():
        lfc_col, _, padj_col = _detect_edger_columns(df)
        up = int(((df[padj_col] < fdr_thresh) & (df[lfc_col] >= lfc_thresh)).sum())
        down = int(((df[padj_col] < fdr_thresh) & (df[lfc_col] <= -lfc_thresh)).sum())
        rows.append({"cell_type": cell_type, "Up": up, "Down": down})

    summary_df = pd.DataFrame(rows).set_index("cell_type")

    fig, ax = plt.subplots(figsize=(9, 6), facecolor="white")
    summary_df.plot(
        kind="bar",
        ax=ax,
        color={"Up": "#d62728", "Down": "#1f77b4"},
        edgecolor="black",
        rot=30,
        width=0.7,
    )

    ax.set_title(
        f"Differentially Expressed Genes per Cell Type ({cohort_label})\n"
        f"[FDR < {fdr_thresh}, |log2FC| >= {lfc_thresh}]"
    )
    ax.set_xlabel("Cell Type (HMBA_Group_name)")
    ax.set_ylabel("Number of DE Genes")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for container in ax.containers:
        ax.bar_label(container, padding=3)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def run_edger_and_export_rnk(
    pdata: ad.AnnData,
    cohort_label: str,
    output_dir: Path,
    plots_dir: Path,
    config: dict[str, Any],
    min_subjects_group: int = 5,
) -> dict[str, pd.DataFrame]:
    """Runs EdgeR across cell types, exporting .csv tables, .rnk GSEA files, and Volcano plots."""
    cell_types = pdata.obs["HMBA_Group_name"].unique()
    results_by_celltype: dict[str, pd.DataFrame] = {}

    for cell_type in cell_types:
        LOGGER.info("--- Differential Expression Analysis (%s) for: %s ---", cohort_label, cell_type)

        # 1. Isolate rows for the specific cell type
        pdata_sub = pdata[pdata.obs["HMBA_Group_name"] == cell_type].copy()

        # --- METADATA CLEANUP ---
        required_cols = ["ASAP_subject_id", "condition_id", "HMBA_Group_name", "age", "sex"]
        pdata_sub.obs = pdata_sub.obs[
            [c for c in required_cols if c in pdata_sub.obs.columns]
        ].copy()

        pdata_sub.obs["age"] = pd.to_numeric(pdata_sub.obs["age"], errors="coerce").astype(float)

        # Rimuovi campioni con NaN in condition_id o age prima di contare i soggetti
        valid_mask = pdata_sub.obs["condition_id"].notna() & pdata_sub.obs["age"].notna()
        if "sex" in pdata_sub.obs.columns:
            valid_mask &= pdata_sub.obs["sex"].notna()
        pdata_sub = pdata_sub[valid_mask].copy()

        # Pulisci eventuali categorie inutilizzate che bloccano model.matrix in R
        for col in ["condition_id", "sex", "HMBA_Group_name"]:
            if col in pdata_sub.obs.columns and isinstance(pdata_sub.obs[col].dtype, pd.CategoricalDtype):
                pdata_sub.obs[col] = pdata_sub.obs[col].cat.remove_unused_categories()

        # 2. Ensure BOTH conditions meet the minimum sample threshold
        counts_per_cond = pdata_sub.obs["condition_id"].value_counts()
        count_control = int(counts_per_cond.get("Control", 0))
        count_pd = int(counts_per_cond.get("PD", 0))

        if count_control < min_subjects_group or count_pd < min_subjects_group:
            LOGGER.warning(
                "Insufficient samples for %s in %s (Control: %d, PD: %d, min required: %d). Skipping...",
                cell_type,
                cohort_label,
                count_control,
                count_pd,
                min_subjects_group,
            )
            continue

        LOGGER.info(
            "Found %d Control samples and %d PD samples for %s",
            count_control,
            count_pd,
            cell_type,
        )

        # 3. Filter out lowly expressed genes for this specific cell type
        dc.pp.filter_by_expr(pdata_sub, group="condition_id")
        if pdata_sub.n_vars == 0:
            LOGGER.warning("No genes passed filter_by_expr for %s in %s. Skipping...", cell_type, cohort_label)
            continue

        # 4. Build design formula dynamically based on non-constant covariates
        covariates = []
        if pdata_sub.obs["age"].nunique() > 1:
            covariates.append("age")
        if "sex" in pdata_sub.obs.columns and pdata_sub.obs["sex"].nunique() > 1:
            covariates.append("sex")
        covariates.append("condition_id")
        design_formula = "~" + " + ".join(covariates)

        # 5. Fit EdgeR model with design_formula and test PD vs Control contrast
        edgr = pt.tl.EdgeR(pdata_sub, design=design_formula)
        edgr.fit()
        edger_results = edgr.test_contrasts(
            edgr.contrast(
                column="condition_id",
                baseline="Control",
                group_to_compare="PD",
            )
        )

        # 6. Export CSV table
        if "variable" in edger_results.columns:
            edger_results = edger_results.set_index("variable")

        safe_cell_name = _sanitize_name(cell_type)
        csv_file_path = output_dir / f"edgeR_{cohort_label}_PDvsControl_{safe_cell_name}.csv"
        edger_results.to_csv(csv_file_path)
        LOGGER.info("Saved CSV: %s", csv_file_path)

        # 7. Export dynamic .rnk file for GSEA / WebGestalt
        rnk_file_path = output_dir / f"webgestalt_data_{cohort_label}_{safe_cell_name}.rnk"
        prepare_gsea_rnk(edger_results, rnk_file_path)
        LOGGER.info("Saved GSEA .rnk file: %s", rnk_file_path)

        # 8. Save Volcano Plot for this cell type
        volcano_path = plots_dir / f"step03_volcano_{cohort_label}_{safe_cell_name}.png"
        save_plot_volcano(
            df_res=edger_results,
            cohort_label=cohort_label,
            cell_type=str(cell_type),
            output_path=volcano_path,
            config=config,
        )

        results_by_celltype[str(cell_type)] = edger_results

    # 9. Save DE Summary Barplot across all analyzed cell types for this cohort
    deg_summary_path = plots_dir / f"step03_deg_summary_{cohort_label}.png"
    save_plot_deg_summary(
        results_by_celltype=results_by_celltype,
        cohort_label=cohort_label,
        output_path=deg_summary_path,
        config=config,
    )

    return results_by_celltype


def format_pipeline_step03_summary(
    *,
    dataset_path: str,
    config_path: str,
    sex_group: str,
    combined: bool,
    output_dir: str,
    plots_dir: str,
    report_path: str,
) -> None:
    LOGGER.info(
        "Pipeline Step03 summary:\n"
        f"- Dopaminergic Dataset: {dataset_path}\n"
        f"- Config File: {config_path}\n"
        f"- Sex Group: {sex_group} (combined={combined})\n"
        f"- Output Directory (CSV & RNK): {output_dir}\n"
        f"- Plots Directory: {plots_dir}\n"
        f"- Report Path: {report_path}"
    )


def build_step03_parser(default_repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step03 Differential Expression Analysis (Pseudobulk + EdgeR)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    processed_dir = default_repo_root / "pipeline" / "data" / "processed"
    plots_dir = default_repo_root / "pipeline" / "plots"
    reports_dir = default_repo_root / "pipeline" / "reports"

    parser.add_argument(
        "--config",
        default=str(default_repo_root / "pipeline" / "config" / "step03.yml"),
        help="YAML config file for step03 tunables.",
    )
    parser.add_argument(
        "--dataset-path",
        default=str(processed_dir / CANONICAL_NAMES["dataset_dopaminergic"]),
        help="Path to the dopaminergic dataset file (.h5ad).",
    )
    parser.add_argument(
        "--sex-group",
        choices=["male", "female", "both"],
        default="both",
        help="Target cohort for DEA: 'male', 'female', or 'both'.",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help=(
            "When used with '--sex-group both', runs the analysis on males and females "
            "pooled together in a single pseudobulk instead of separately."
        ),
    )
    parser.add_argument(
        "--min-cells",
        type=int,
        default=None,
        help="Minimum number of cells per sample in pseudobulk (overrides config).",
    )
    parser.add_argument(
        "--min-counts",
        type=int,
        default=None,
        help="Minimum number of counts per sample in pseudobulk (overrides config).",
    )
    parser.add_argument(
        "--min-subjects-group",
        type=int,
        default=None,
        help="Minimum number of subjects per condition (Control/PD) in EdgeR (overrides config).",
    )
    parser.add_argument(
        "--qc-plot-path",
        default=str(plots_dir / "step03_pseudobulk_qc.png"),
        help="Base filename or path for the pseudobulk QC plot (cohort label is appended automatically).",
    )
    parser.add_argument("--reports-dir", default=str(reports_dir))
    parser.add_argument("--plots-dir", default=str(plots_dir))
    parser.add_argument("--output-dir", default=str(processed_dir / "step03_edger"))
    parser.add_argument("--logging-level", default="INFO")
    return parser


def run_step03(args: argparse.Namespace) -> None:
    ro.r('library(edgeR)')
    configure_logging(args.logging_level)
    config_path = Path(args.config)
    config = _resolve_step03_config(config_path)

    min_cells = (
        args.min_cells if args.min_cells is not None else int(config["min_cells_pseudobulk"])
    )
    min_counts = (
        args.min_counts if args.min_counts is not None else int(config["min_counts_pseudobulk"])
    )
    min_subjects = (
        args.min_subjects_group
        if args.min_subjects_group is not None
        else int(config["min_subjects_group"])
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    qc_plot_path = Path(args.qc_plot_path)

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    adata = sc.read_h5ad(dataset_path)

    # Determine which cohorts to run based on CLI arguments
    cohorts_to_run: list[tuple[str, ad.AnnData]] = []
    if args.sex_group == "male":
        cohorts_to_run.append(("MALES", adata[adata.obs["sex"] == "Male"].copy()))
    elif args.sex_group == "female":
        cohorts_to_run.append(("FEMALES", adata[adata.obs["sex"] == "Female"].copy()))
    elif args.sex_group == "both":
        if args.combined:
            cohorts_to_run.append(("ALL", adata.copy()))
        else:
            cohorts_to_run.append(("MALES", adata[adata.obs["sex"] == "Male"].copy()))
            cohorts_to_run.append(("FEMALES", adata[adata.obs["sex"] == "Female"].copy()))

    fdr_thresh = float(config["fdr_threshold"])
    lfc_thresh = float(config["logfc_threshold"])

    report_summary: dict[str, Any] = {
        "config_path": str(config_path),
        "dataset_path": str(dataset_path),
        "sex_group": args.sex_group,
        "combined": bool(args.combined),
        "min_cells_pseudobulk": min_cells,
        "min_counts_pseudobulk": min_counts,
        "min_subjects_group": min_subjects,
        "fdr_threshold": fdr_thresh,
        "logfc_threshold": lfc_thresh,
        "most_significant_dea": int(config["most_significant_dea"]),
        "cohorts": {},
    }

    for cohort_label, sub_adata in cohorts_to_run:
        cohort_qc_plot_path = build_tagged_path(qc_plot_path.name, cohort_label, default_dir=qc_plot_path.parent)
        pdata = make_pseudobulk(
            adata=sub_adata,
            cohort_label=cohort_label,
            min_cells=min_cells,
            min_counts=min_counts,
            qc_plot_path=cohort_qc_plot_path,
            dpi=int(config["dpi_figures"]),
        )

        edger_results = run_edger_and_export_rnk(
            pdata=pdata,
            cohort_label=cohort_label,
            output_dir=out_dir,
            plots_dir=plots_dir,
            config=config,
            min_subjects_group=min_subjects,
        )

        cohort_stats: dict[str, Any] = {}
        for cell_type, df_res in edger_results.items():
            lfc_col, _, padj_col = _detect_edger_columns(df_res)
            cohort_stats[cell_type] = {
                "tested_genes": int(df_res.shape[0]),
                "significant_fdr": int((df_res[padj_col] < fdr_thresh).sum()),
                "upregulated": int(
                    ((df_res[padj_col] < fdr_thresh) & (df_res[lfc_col] >= lfc_thresh)).sum()
                ),
                "downregulated": int(
                    ((df_res[padj_col] < fdr_thresh) & (df_res[lfc_col] <= -lfc_thresh)).sum()
                ),
            }
        report_summary["cohorts"][cohort_label] = cohort_stats

    report_path = reports_dir / "step03_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report_summary, handle, indent=2)

    format_pipeline_step03_summary(
        dataset_path=str(dataset_path),
        config_path=str(config_path),
        sex_group=args.sex_group,
        combined=bool(args.combined),
        output_dir=str(out_dir),
        plots_dir=str(plots_dir),
        report_path=str(report_path),
    )