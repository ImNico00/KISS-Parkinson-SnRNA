from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse

from pipeline.src.schemas import CANONICAL_NAMES
from pipeline.src.utils import build_tagged_path, configure_logging, load_yaml_file

LOGGER = logging.getLogger(__name__)

DEFAULT_STEP02_CONFIG: dict[str, Any] = {
    "dpi_figures": 300,
    "age_bins": [0, 60, 70, 80, 120],
    "age_labels": ["<60", "60-69", "70-79", "80+"],
}


def _age_stats(series: pd.Series) -> str:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return "N/A"
    mean_age = clean.mean()
    std_age = clean.std()
    min_age = clean.min()
    max_age = clean.max()
    if pd.isna(std_age):
        return f"{mean_age:.1f} ({min_age:.0f}-{max_age:.0f})"
    return f"{mean_age:.1f} ± {std_age:.1f} ({min_age:.0f}-{max_age:.0f})"


def _extract_patients_df(adata: ad.AnnData) -> pd.DataFrame:
    """Extracts a deduplicated subject-level table based on ASAP_subject_id."""
    df_obs = adata.obs[["ASAP_subject_id", "condition_id", "sex", "age"]].copy()
    df_patients = (
        df_obs.groupby("ASAP_subject_id", observed=True).first().reset_index()
    )
    df_patients["age"] = pd.to_numeric(df_patients["age"], errors="coerce")
    return df_patients


def show_demographic_summary(adata: ad.AnnData) -> None:
    df_patients = _extract_patients_df(adata)

    summary = (
        df_patients.groupby(["condition_id", "sex"], observed=True)
        .agg(
            Number_of_Subjects=("ASAP_subject_id", "count"),
            Age_Mean_Std_Range=("age", _age_stats),
        )
        .reset_index()
    )

    LOGGER.info("===============================================================")
    LOGGER.info("      DEMOGRAPHIC SUMMARY OF THE HUMAN DOPAMINERGIC COHORT     ")
    LOGGER.info("===============================================================")
    LOGGER.info("\n%s", summary.to_string(index=False, justify="center"))
    LOGGER.info("===============================================================\n")

    total_subjects = df_patients.shape[0]
    total_males = int((df_patients["sex"] == "Male").sum())
    total_females = int((df_patients["sex"] == "Female").sum())
    overall_age = _age_stats(df_patients["age"])

    LOGGER.info("TOTAL SUBJECTS: %d", total_subjects)
    LOGGER.info(" -> Males: %d", total_males)
    LOGGER.info(" -> Females: %d", total_females)
    LOGGER.info("OVERALL AGE: %s", overall_age)


def save_plot_umap(
    config: dict[str, Any],
    adata: ad.AnnData,
    subclusters_figure_path: Path | str,
) -> None:
    dpi = int(config["dpi_figures"])
    sc.set_figure_params(dpi=dpi // 2, figsize=(8, 8), facecolor="white")
    sc.pl.umap(
        adata,
        color=["HMBA_Group_name"],
        ncols=3,
        wspace=0.8,
        frameon=False,
        show=False,
    )
    plt.savefig(subclusters_figure_path, dpi=dpi, bbox_inches="tight")
    plt.close("all")


def save_plot_dotplot_kiss1r(
    config: dict[str, Any],
    adata: ad.AnnData,
    kiss1r_dotplot_figure_path: Path | str,
) -> None:
    target_genes = [g for g in ["KISS1R"] if g in adata.var_names]
    if not target_genes:
        LOGGER.warning(
            "None of the target genes %s were found in adata.var_names. Skipping dotplot.",
            ["KISS1R"],
        )
        return

    dpi = int(config["dpi_figures"])
    sc.set_figure_params(dpi=dpi // 2, figsize=(8, 8), facecolor="white")
    sc.pl.dotplot(
        adata,
        var_names=target_genes,
        groupby="sex",
        show=False,
    )
    plt.savefig(kiss1r_dotplot_figure_path, dpi=dpi, bbox_inches="tight")
    plt.close("all")


def save_plot_dotplot_kiss1r_per_group(
    config: dict[str, Any],
    adata: ad.AnnData,
    kiss1r_dotplot_per_group_figure_path: Path | str,
) -> None:
    target_genes = [g for g in ["KISS1R"] if g in adata.var_names]
    if not target_genes:
        LOGGER.warning(
            "None of the target genes %s were found in adata.var_names. Skipping per-group dotplot.",
            ["KISS1R"],
        )
        return

    dpi = int(config["dpi_figures"])
    for sex in adata.obs["sex"].dropna().unique():
        group_mask = adata.obs["sex"] == sex
        group_adata = adata[group_mask].copy()
        if group_adata.n_obs == 0:
            continue

        if isinstance(group_adata.obs["HMBA_Group_name"].dtype, pd.CategoricalDtype):
            group_adata.obs["HMBA_Group_name"] = (
                group_adata.obs["HMBA_Group_name"].cat.remove_unused_categories()
            )

        group_figure_path = build_tagged_path(
            kiss1r_dotplot_per_group_figure_path, sex
        )

        sc.set_figure_params(dpi=dpi // 2, figsize=(8, 8), facecolor="white")
        sc.pl.dotplot(
            group_adata,
            var_names=target_genes,
            groupby="HMBA_Group_name",
            show=False,
        )
        plt.savefig(group_figure_path, dpi=dpi, bbox_inches="tight")
        plt.close("all")


def save_plot_subjects_by_condition_sex(
    config: dict[str, Any],
    adata: ad.AnnData,
    condition_sex_figure_path: Path | str,
) -> None:
    """Saves a barplot showing the percentage of subjects stratified by condition_id and sex."""
    df_patients = _extract_patients_df(adata)
    if df_patients.empty:
        LOGGER.warning("No subjects found for the condition_id vs sex plot.")
        return

    # Compute percentage over total cohort subjects
    ct = (
        pd.crosstab(
            df_patients["condition_id"],
            df_patients["sex"],
            normalize="all",
        )
        * 100.0
    )

    dpi = int(config["dpi_figures"])
    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    ct.plot(kind="bar", ax=ax, rot=0, width=0.7, edgecolor="black")

    ax.set_title("Subjects Distribution by Condition and Sex (%)")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Percentage of Total Subjects (%)")
    ax.legend(title="Sex", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=3)

    ax.set_ylim(0, max(100.0, ct.to_numpy().max() * 1.15))
    fig.savefig(condition_sex_figure_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_plot_subjects_by_condition_age(
    config: dict[str, Any],
    adata: ad.AnnData,
    condition_age_figure_path: Path | str,
) -> None:
    """Saves a barplot showing the percentage of subjects stratified by condition_id and age group."""
    df_patients = _extract_patients_df(adata).dropna(subset=["age"])
    if df_patients.empty:
        LOGGER.warning("No valid age data found for the condition_id vs age plot.")
        return

    bins = config.get("age_bins", DEFAULT_STEP02_CONFIG["age_bins"])
    labels = config.get("age_labels", DEFAULT_STEP02_CONFIG["age_labels"])

    df_patients["age_group"] = pd.cut(
        df_patients["age"],
        bins=bins,
        labels=labels,
        right=False,
    )

    ct = (
        pd.crosstab(
            df_patients["condition_id"],
            df_patients["age_group"],
            normalize="all",
            dropna=False,
        )
        * 100.0
    )

    dpi = int(config["dpi_figures"])
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="white")
    ct.plot(kind="bar", ax=ax, rot=0, width=0.75, edgecolor="black")

    ax.set_title("Subjects Distribution by Condition and Age Group (%)")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Percentage of Total Subjects (%)")
    ax.legend(title="Age Group", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=3, fontsize=9)

    ax.set_ylim(0, max(100.0, ct.to_numpy().max() * 1.15))
    fig.savefig(condition_age_figure_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _resolve_step02_config(config_path: Path) -> dict[str, Any]:
    config = DEFAULT_STEP02_CONFIG.copy()
    if config_path.exists():
        loaded = load_yaml_file(config_path) or {}
        step02_cfg = loaded.get("step02", {})
        if step02_cfg and not isinstance(step02_cfg, dict):
            raise ValueError(f"Expected mapping under 'step02' in config: {config_path}")
        config.update({k: v for k, v in step02_cfg.items() if v is not None})
    else:
        LOGGER.warning("Config file not found (%s), using default values.", config_path)
    return config


def compute_subgroup_stats(adata: ad.AnnData, mask: pd.Series) -> dict[str, int | None]:
    """Computes summary statistics on an AnnData view without allocating full copies in RAM."""
    sub = adata[mask]
    if sub.n_obs == 0:
        return {
            "number_of_subjects": 0,
            "max_age": None,
            "min_age": None,
            "cells_count": 0,
            "total_genes_prefiltering": int(adata.n_vars),
            "total_genes_postfiltering": 0,
        }

    ages = pd.to_numeric(sub.obs["age"], errors="coerce").dropna()
    gene_sums = np.asarray(sub.X.sum(axis=0)).ravel()
    active_genes = int((gene_sums > 0).sum())

    return {
        "number_of_subjects": int(sub.obs["ASAP_subject_id"].nunique()),
        "max_age": int(ages.max()) if not ages.empty else None,
        "min_age": int(ages.min()) if not ages.empty else None,
        "cells_count": int(sub.n_obs),
        "total_genes_prefiltering": int(adata.n_vars),
        "total_genes_postfiltering": active_genes,
    }


def compute_gene_pct(
    adata: ad.AnnData,
    gene: str,
    mask: pd.Series | None = None,
) -> float:
    """Returns the percentage of cells with expression > 0 for the specified gene."""
    if gene not in adata.var_names:
        LOGGER.warning("Gene '%s' not found in adata.var_names.", gene)
        return 0.0

    sub_gene = adata[mask, gene] if mask is not None else adata[:, gene]
    if sub_gene.n_obs == 0:
        return 0.0

    x = sub_gene.X
    expr = x.toarray() if issparse(x) else np.asarray(x)
    return float((expr > 0).mean() * 100.0)


def format_pipeline_step02_summary(
    *,
    dataset_path: str,
    config_path: str,
    subclusters_figure_path: str,
    kiss1r_dotplot_figure_path: str,
    kiss1r_dotplot_per_group_figure_path: str,
    condition_sex_figure_path: str,
    condition_age_figure_path: str,
    output_dir: str,
    report_path: str,
) -> None:
    LOGGER.info(
        "Pipeline Step02 summary:\n"
        f"- Dopaminergic Dataset: {dataset_path}\n"
        f"- Config File: {config_path}\n"
        f"- Subclusters Figure: {subclusters_figure_path}\n"
        f"- KISS1R Dotplot Figure: {kiss1r_dotplot_figure_path}\n"
        f"- KISS1R Dotplot Per Group Base Path: {kiss1r_dotplot_per_group_figure_path}\n"
        f"- Condition/Sex Figure: {condition_sex_figure_path}\n"
        f"- Condition/Age Figure: {condition_age_figure_path}\n"
        f"- Output Directory: {output_dir}\n"
        f"- Report Path: {report_path}"
    )


def build_step02_parser(default_repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step02 Data Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    processed_dir = default_repo_root / "pipeline" / "data" / "processed"
    plots_dir = default_repo_root / "pipeline" / "plots"
    reports_dir = default_repo_root / "pipeline" / "reports"

    parser.add_argument(
        "--config",
        default=str(default_repo_root / "pipeline" / "config" / "step02.yml"),
        help="YAML config file for step02 tunables.",
    )
    parser.add_argument(
        "--dataset-path",
        default=str(processed_dir / CANONICAL_NAMES["dataset_dopaminergic"]),
        help="Path to the dopaminergic dataset file.",
    )
    parser.add_argument(
        "--subclusters-figure-path",
        default=str(plots_dir / "step02_umap.png"),
        help="Path to save the UMAP subclusters figure.",
    )
    parser.add_argument(
        "--kiss1r-dotplot-figure-path",
        default=str(plots_dir / "step02_kiss1r_dotplot.png"),
        help="Path to save the KISS1R dotplot figure.",
    )
    parser.add_argument(
        "--kiss1r-dotplot-per-group-figure-path",
        default=str(plots_dir / "step02_kiss1r_dotplot_per_group.png"),
        help="Base path to save the KISS1R dotplot figure per group.",
    )
    parser.add_argument(
        "--condition-sex-figure-path",
        default=str(plots_dir / "step02_subjects_condition_sex.png"),
        help="Path to save the subjects percentage by condition and sex figure.",
    )
    parser.add_argument(
        "--condition-age-figure-path",
        default=str(plots_dir / "step02_subjects_condition_age.png"),
        help="Path to save the subjects percentage by condition and age figure.",
    )
    parser.add_argument("--reports-dir", default=str(reports_dir))
    parser.add_argument("--plots-dir", default=str(plots_dir))
    parser.add_argument("--output-dir", default=str(processed_dir))
    parser.add_argument("--logging-level", default="INFO")
    return parser


def run_step02(args: argparse.Namespace) -> None:
    configure_logging(args.logging_level)
    config_path = Path(args.config)
    config = _resolve_step02_config(config_path)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    subclusters_figure_path = Path(args.subclusters_figure_path)
    kiss1r_dotplot_figure_path = Path(args.kiss1r_dotplot_figure_path)
    kiss1r_dotplot_per_group_figure_path = Path(args.kiss1r_dotplot_per_group_figure_path)
    condition_sex_figure_path = Path(args.condition_sex_figure_path)
    condition_age_figure_path = Path(args.condition_age_figure_path)

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    adata = sc.read_h5ad(dataset_path)
    show_demographic_summary(adata)

    mask_male = adata.obs["sex"] == "Male"
    mask_female = adata.obs["sex"] == "Female"
    mask_ctrl = adata.obs["condition_id"] == "Control"
    mask_pd = adata.obs["condition_id"] == "PD"

    mask_male_ctrl = mask_male & mask_ctrl
    mask_male_pd = mask_male & mask_pd
    mask_female_ctrl = mask_female & mask_ctrl
    mask_female_pd = mask_female & mask_pd

    save_plot_umap(
        config=config,
        adata=adata,
        subclusters_figure_path=subclusters_figure_path,
    )
    save_plot_dotplot_kiss1r(
        config=config,
        adata=adata,
        kiss1r_dotplot_figure_path=kiss1r_dotplot_figure_path,
    )
    save_plot_subjects_by_condition_sex(
        config=config,
        adata=adata,
        condition_sex_figure_path=condition_sex_figure_path,
    )
    save_plot_subjects_by_condition_age(
        config=config,
        adata=adata,
        condition_age_figure_path=condition_age_figure_path,
    )
    save_plot_dotplot_kiss1r_per_group(
        config=config,
        adata=adata,
        kiss1r_dotplot_per_group_figure_path=kiss1r_dotplot_per_group_figure_path,
    )

    report = {
        "config_path": str(config_path),
        "total_subjects": int(adata.obs["ASAP_subject_id"].nunique()),
        "total_cells": int(adata.n_obs),
        "total_genes": int(adata.n_vars),
        "age_labels": config["age_labels"],
        "age_bins": config["age_bins"],
        "male_subjects": {
            "overall": compute_subgroup_stats(adata, mask_male),
            "controls": compute_subgroup_stats(adata, mask_male_ctrl),
            "parkinsons": compute_subgroup_stats(adata, mask_male_pd),
        },
        "female_subjects": {
            "overall": compute_subgroup_stats(adata, mask_female),
            "controls": compute_subgroup_stats(adata, mask_female_ctrl),
            "parkinsons": compute_subgroup_stats(adata, mask_female_pd),
        },
        "KISS1R_expression": {
            "total_pct": compute_gene_pct(adata, "KISS1R"),
            "male_pct": {
                "overall": compute_gene_pct(adata, "KISS1R", mask_male),
                "controls": compute_gene_pct(adata, "KISS1R", mask_male_ctrl),
                "parkinsons": compute_gene_pct(adata, "KISS1R", mask_male_pd),
            },
            "female_pct": {
                "overall": compute_gene_pct(adata, "KISS1R", mask_female),
                "controls": compute_gene_pct(adata, "KISS1R", mask_female_ctrl),
                "parkinsons": compute_gene_pct(adata, "KISS1R", mask_female_pd),
            },
        },
    }

    report_path = reports_dir / "step02_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    format_pipeline_step02_summary(
        dataset_path=str(dataset_path),
        config_path=str(config_path),
        subclusters_figure_path=str(subclusters_figure_path),
        kiss1r_dotplot_figure_path=str(kiss1r_dotplot_figure_path),
        kiss1r_dotplot_per_group_figure_path=str(kiss1r_dotplot_per_group_figure_path),
        condition_sex_figure_path=str(condition_sex_figure_path),
        condition_age_figure_path=str(condition_age_figure_path),
        output_dir=str(out_dir),
        report_path=str(report_path),
    )