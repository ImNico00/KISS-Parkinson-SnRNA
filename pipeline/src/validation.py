from __future__ import annotations

import argparse
import ast
import itertools
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
import scanpy as sc
import scipy.sparse as sp
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

DEFAULT_STEP06_CONFIG: dict[str, Any] = {
    "dpi_figures": 300,
    "inherit_step04_params": True,
    "consensus_mode": "step04_best",  # "step04_best", "step04_optimize", or "custom"
    "min_cells_pseudobulk": 20,
    "target_sum_norm": 10000.0,
    "comparison_mode": "proxy",  # "proxy", "kiss1r", or "condition"
    "kiss1r_gene": "KISS1R",
    "gnaq_gene": "GNAQ",
    "proxy_genes": ["CALB2", "TH", "S100B", "GNRH1"],
    "min_active_proxies": 3,
    "decoupler_methods": ["ulm", "waggr", "viper", "aucell", "gsea", "gsva", "mlm", "zscore"],
    "min_combo_size": 3,
    "tmin_targets": 6,
    "fdr_threshold": 0.05,
    "diff_threshold": 0.25,
    "pathway_database": "progeny",  # "progeny", "hallmark", or "carnival_only"
    "progeny_top_genes": 500,
    "min_pathway_carnival_overlap": 1,
    "correlation_method": "spearman",  # "spearman" or "pearson"
    "top_plot_items": 20,
}


def _resolve_step06_config(
    config_path: Path,
    step04_report_path: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    """
    Loads step06.yml and automatically inherits decoupler/stratification parameters
    and per-cohort best_combo from step04_report.json when inherit_step04_params=True.
    """
    config = DEFAULT_STEP06_CONFIG.copy()
    step06_explicit_keys: set[str] = set()

    if config_path.exists():
        loaded = load_yaml_file(config_path) or {}
        step06_cfg = loaded.get("step06", {})
        if step06_cfg and not isinstance(step06_cfg, dict):
            raise ValueError(f"Expected mapping under 'step06' in config: {config_path}")
        for k, v in step06_cfg.items():
            if v is not None:
                config[k] = v
                step06_explicit_keys.add(k)
    else:
        LOGGER.warning("Config file not found (%s), using default values.", config_path)

    step04_best_combos: dict[str, str] = {}
    if step04_report_path.exists():
        with step04_report_path.open("r", encoding="utf-8") as handle:
            rep04 = json.load(handle)

        if bool(config.get("inherit_step04_params", True)):
            p04 = rep04.get("parameters", {})
            inherited_map = {
                "min_cells_pseudobulk": p04.get("min_cells_pseudobulk"),
                "target_sum_norm": p04.get("target_sum_norm"),
                "comparison_mode": rep04.get("stratification_mode"),
                "kiss1r_gene": p04.get("kiss1r_gene"),
                "gnaq_gene": p04.get("gnaq_gene"),
                "proxy_genes": p04.get("proxy_genes"),
                "min_active_proxies": p04.get("min_active_proxies"),
                "decoupler_methods": p04.get("decoupler_methods"),
                "min_combo_size": p04.get("min_combo_size"),
                "tmin_targets": p04.get("tmin_targets"),
                "fdr_threshold": p04.get("fdr_threshold"),
                "diff_threshold": p04.get("diff_threshold"),
            }
            for k, val in inherited_map.items():
                # Inherit from Step 04 unless explicitly overridden in step06.yml
                if val is not None and k not in step06_explicit_keys:
                    config[k] = val

            LOGGER.info(
                "Inherited Step 04 parameters from %s (tmin_targets=%d, comparison_mode=%s).",
                step04_report_path,
                int(config["tmin_targets"]),
                config["comparison_mode"],
            )

        for cohort_label, cdata in rep04.get("cohorts", {}).items():
            best_combo = cdata.get("decoupler_consensus", {}).get("best_combo")
            if best_combo:
                step04_best_combos[cohort_label.upper()] = str(best_combo)
                cohort_key = str(cdata.get("cohort_key", "")).upper()
                if cohort_key:
                    step04_best_combos[cohort_key] = str(best_combo)
    else:
        LOGGER.warning(
            "Step 04 report not found at %s; using step06 config parameters directly.",
            step04_report_path,
        )

    config["proxy_genes"] = [str(g) for g in _ensure_list(config["proxy_genes"])]
    config["decoupler_methods"] = [str(m) for m in _ensure_list(config["decoupler_methods"])]
    return config, step04_best_combos


def _infer_cohort_from_label(network_label: str, default_cohort: str = "all") -> str:
    """Infers the target subject cohort from a Step 05 network label."""
    upper = network_label.upper()
    if "CONTROL_FEMALE" in upper:
        return "control_female"
    if "CONTROL_MALE" in upper:
        return "control_male"
    if "PD_FEMALE" in upper:
        return "pd_female"
    if "PD_MALE" in upper:
        return "pd_male"
    if "FEMALE" in upper:
        return "female"
    if "MALE" in upper:
        return "male"
    return default_cohort


def _subset_adata_by_cohort(adata: ad.AnnData, cohort_key: str) -> tuple[str, ad.AnnData]:
    """Subsets AnnData according to the requested cohort key."""
    sex_s = adata.obs["sex"].astype(str)
    cond_s = adata.obs["condition_id"].astype(str)

    cohort_map: dict[str, tuple[str, pd.Series]] = {
        "control_male": ("CONTROL_MALES", (sex_s == "Male") & (cond_s == "Control")),
        "control_female": ("CONTROL_FEMALES", (sex_s == "Female") & (cond_s == "Control")),
        "pd_male": ("PD_MALES", (sex_s == "Male") & (cond_s == "PD")),
        "pd_female": ("PD_FEMALES", (sex_s == "Female") & (cond_s == "PD")),
        "male": ("MALES", sex_s == "Male"),
        "female": ("FEMALES", sex_s == "Female"),
        "all": ("ALL", pd.Series(True, index=adata.obs.index)),
    }

    key_lower = cohort_key.lower()
    if key_lower not in cohort_map:
        raise ValueError(f"Unsupported cohort key '{cohort_key}'.")

    label, mask = cohort_map[key_lower]
    return label, adata[mask].copy()


def prepare_subject_pseudobulk(
    adata: ad.AnnData,
    cohort_label: str,
    config: dict[str, Any],
    min_cells: int,
    comparison_mode: str,
) -> tuple[ad.AnnData, str, str, str, dict[str, Any]]:
    """
    Aggregates single-cell data into subject-level pseudobulk profiles using the exact
    Step 04 normalization and stratification logic ('proxy', 'kiss1r', or 'condition').
    """
    pdata = dc.pp.pseudobulk(
        adata,
        sample_col="ASAP_subject_id",
        groups_col=None,
        mode="sum",
    )
    n_pre = int(pdata.n_obs)
    dc.pp.filter_samples(pdata, min_cells=min_cells)
    n_post = int(pdata.n_obs)

    if n_post == 0:
        return pdata, "PROXY_status", "High", "Low", {
            "subjects_prefilter": n_pre,
            "subjects_postfilter": 0,
        }

    pdata_norm = pdata.copy()
    sc.pp.normalize_total(pdata_norm, target_sum=float(config["target_sum_norm"]))
    sc.pp.log1p(pdata_norm)

    if comparison_mode in {"proxy", "kiss1r"}:
        kiss1r_gene = str(config["kiss1r_gene"])
        kiss1r_pos = (
            _extract_1d_expr(pdata_norm, kiss1r_gene) > 0
            if kiss1r_gene in pdata_norm.var_names
            else np.zeros(n_post, dtype=bool)
        )

        if comparison_mode == "proxy":
            gnaq_gene = str(config["gnaq_gene"])
            gnaq_pos = (
                _extract_1d_expr(pdata_norm, gnaq_gene) > 0
                if gnaq_gene in pdata_norm.var_names
                else np.zeros(n_post, dtype=bool)
            )
            proxy_genes = [g for g in config["proxy_genes"] if g in pdata_norm.var_names]
            if proxy_genes:
                proxy_count = np.sum(
                    [(_extract_1d_expr(pdata_norm, g) > 0).astype(int) for g in proxy_genes],
                    axis=0,
                )
            else:
                proxy_count = np.zeros(n_post, dtype=int)
            is_high = kiss1r_pos | (gnaq_pos & (proxy_count >= int(config["min_active_proxies"])))
        else:
            is_high = kiss1r_pos

        pdata_norm.obs["PROXY_status"] = np.where(is_high, "High", "Low")
        group_col, pos_label, neg_label = "PROXY_status", "High", "Low"
    elif comparison_mode == "condition":
        pdata_norm.obs["PROXY_status"] = pdata_norm.obs["condition_id"].astype(str)
        group_col, pos_label, neg_label = "PROXY_status", "PD", "Control"
    else:
        raise ValueError(f"Unsupported comparison_mode: {comparison_mode}")

    n_pos = int((pdata_norm.obs[group_col] == pos_label).sum())
    n_neg = int((pdata_norm.obs[group_col] == neg_label).sum())

    stats_dict = {
        "cohort_label": cohort_label,
        "comparison_mode": comparison_mode,
        "group_column": group_col,
        "positive_group": pos_label,
        "negative_group": neg_label,
        "subjects_prefilter": n_pre,
        "subjects_postfilter": n_post,
        f"subjects_{pos_label}": n_pos,
        f"subjects_{neg_label}": n_neg,
    }
    return pdata_norm, group_col, pos_label, neg_label, stats_dict


def _discover_step05_networks(
    networks_dir: Path,
    step05_report_path: Path,
    explicit_nodes: list[str] | None,
    explicit_edges: list[str] | None,
) -> dict[str, dict[str, Any]]:
    """Discovers CARNIVAL node and edge CSV files produced in Step 05."""
    networks: dict[str, dict[str, Any]] = {}

    if explicit_nodes:
        for idx, item in enumerate(explicit_nodes):
            if "=" in item:
                label, raw_path = item.split("=", 1)
                nodes_path = Path(raw_path.strip())
            else:
                nodes_path = Path(item.strip())
                label = re.sub(r"^Carnival_Nodes_", "", nodes_path.stem)

            edges_path = None
            if explicit_edges and idx < len(explicit_edges):
                e_item = explicit_edges[idx]
                edges_path = Path(e_item.split("=", 1)[1].strip() if "=" in e_item else e_item.strip())
            else:
                candidate = nodes_path.with_name(nodes_path.name.replace("Nodes", "Edges"))
                if candidate.exists():
                    edges_path = candidate

            networks[label] = {
                "cohort": _infer_cohort_from_label(label),
                "nodes_csv": nodes_path,
                "edges_csv": edges_path,
            }
        return networks

    if step05_report_path.exists():
        with step05_report_path.open("r", encoding="utf-8") as handle:
            rep05 = json.load(handle)
        for cohort_label, cdata in rep05.get("cohorts", {}).items():
            for state_key, sim in cdata.get("simulations", {}).items():
                if sim.get("status") == "success" and sim.get("nodes_csv"):
                    nodes_p = Path(sim["nodes_csv"])
                    edges_p = Path(sim["edges_csv"]) if sim.get("edges_csv") else None
                    if nodes_p.exists():
                        net_id = f"{cohort_label}_{state_key}"
                        networks[net_id] = {
                            "cohort": _infer_cohort_from_label(cohort_label),
                            "cohort_raw_label": cohort_label,
                            "nodes_csv": nodes_p,
                            "edges_csv": edges_p if (edges_p and edges_p.exists()) else None,
                        }

    if not networks and networks_dir.exists():
        for nodes_p in sorted(networks_dir.glob("Carnival_Nodes_*.csv")):
            net_id = re.sub(r"^Carnival_Nodes_", "", nodes_p.stem)
            edges_p = nodes_p.with_name(nodes_p.name.replace("Carnival_Nodes_", "Carnival_Edges_"))
            networks[net_id] = {
                "cohort": _infer_cohort_from_label(net_id),
                "cohort_raw_label": net_id,
                "nodes_csv": nodes_p,
                "edges_csv": edges_p if edges_p.exists() else None,
            }

    return networks


def evaluate_carnival_nodes_and_edges_on_subjects(
    pdata_norm: ad.AnnData,
    df_nodes: pd.DataFrame,
    df_edges: pd.DataFrame | None,
    group_col: str,
    pos_label: str,
    neg_label: str,
    corr_method: str = "spearman",
    fdr_thresh: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Checks CARNIVAL active nodes and edges against subject-level expression in the .h5ad dataset."""
    mask_pos = (pdata_norm.obs[group_col] == pos_label).to_numpy()
    mask_neg = (pdata_norm.obs[group_col] == neg_label).to_numpy()
    can_test = int(mask_pos.sum()) >= 2 and int(mask_neg.sum()) >= 2

    node_rows: list[dict[str, Any]] = []
    for row in df_nodes.itertuples(index=False):
        node_id = str(row.Node_ID)
        carnival_act = float(row.Activity)
        in_adata = node_id in pdata_norm.var_names

        if not in_adata:
            node_rows.append(
                {
                    "Node_ID": node_id,
                    "CARNIVAL_Activity": carnival_act,
                    "Present_in_H5AD": False,
                    "Mean_Expr_Overall": np.nan,
                    f"Mean_Expr_{pos_label}": np.nan,
                    f"Mean_Expr_{neg_label}": np.nan,
                    "Delta_Expr": np.nan,
                    "P_Value": np.nan,
                    "Concordant_Sign": False,
                }
            )
            continue

        expr = _extract_1d_expr(pdata_norm, node_id)
        mean_all = float(expr.mean())
        mean_pos = float(expr[mask_pos].mean()) if mask_pos.any() else np.nan
        mean_neg = float(expr[mask_neg].mean()) if mask_neg.any() else np.nan
        delta = mean_pos - mean_neg if (not np.isnan(mean_pos) and not np.isnan(mean_neg)) else np.nan

        pval = 1.0
        if can_test:
            _, p = stats.ttest_ind(expr[mask_pos], expr[mask_neg], equal_var=False)
            pval = float(np.nan_to_num(p, nan=1.0))

        concordant = bool(not np.isnan(delta) and (np.sign(delta) == np.sign(carnival_act)))

        node_rows.append(
            {
                "Node_ID": node_id,
                "CARNIVAL_Activity": carnival_act,
                "Present_in_H5AD": True,
                "Mean_Expr_Overall": mean_all,
                f"Mean_Expr_{pos_label}": mean_pos,
                f"Mean_Expr_{neg_label}": mean_neg,
                "Delta_Expr": delta,
                "P_Value": pval,
                "Concordant_Sign": concordant,
            }
        )

    df_node_eval = pd.DataFrame(node_rows)
    if not df_node_eval.empty and df_node_eval["P_Value"].notna().any():
        valid_p = df_node_eval["P_Value"].fillna(1.0).to_numpy()
        _, padj, _, _ = multipletests(valid_p, method="fdr_bh")
        df_node_eval["FDR"] = padj
    else:
        df_node_eval["FDR"] = np.nan

    edge_rows: list[dict[str, Any]] = []
    
    if df_edges is not None and not df_edges.empty:
        df_edges.dropna(subset=["Source", "Target", "Interaction", "Activity"], inplace=True)
        for row in df_edges.itertuples(index=False):
            src = str(row.Source)
            tgt = str(row.Target)
            interaction = int(row.Interaction)
            edge_act = float(row.Activity)

            if src in pdata_norm.var_names and tgt in pdata_norm.var_names and pdata_norm.n_obs >= 3:
                x_src = _extract_1d_expr(pdata_norm, src)
                x_tgt = _extract_1d_expr(pdata_norm, tgt)
                if np.std(x_src) == 0 or np.std(x_tgt) == 0:
                    corr, pval = 0.0, 1.0
                elif corr_method == "pearson":
                    res = stats.pearsonr(x_src, x_tgt)
                    corr, pval = float(res[0]), float(res[1])
                else:
                    res = stats.spearmanr(x_src, x_tgt)
                    corr, pval = float(res[0]), float(res[1])

                corr = float(np.nan_to_num(corr, nan=0.0))
                pval = float(np.nan_to_num(pval, nan=1.0))
                concordant = bool(np.sign(corr) == np.sign(interaction))
                both_present = True
            else:
                corr, pval, concordant, both_present = np.nan, np.nan, False, False

            edge_rows.append(
                {
                    "Source": src,
                    "Interaction": interaction,
                    "Target": tgt,
                    "CARNIVAL_Edge_Activity": edge_act,
                    "Both_Present_in_H5AD": both_present,
                    "Subject_Correlation": corr,
                    "P_Value": pval,
                    "Concordant_Interaction": concordant,
                }
            )

    df_edge_eval = pd.DataFrame(edge_rows)
    if not df_edge_eval.empty and df_edge_eval["P_Value"].notna().any():
        valid_ep = df_edge_eval["P_Value"].fillna(1.0).to_numpy()
        _, epadj, _, _ = multipletests(valid_ep, method="fdr_bh")
        df_edge_eval["FDR"] = epadj
    else:
        df_edge_eval["FDR"] = np.nan

    present_nodes = df_node_eval[df_node_eval["Present_in_H5AD"]]
    present_edges = (
        df_edge_eval[df_edge_eval["Both_Present_in_H5AD"]]
        if not df_edge_eval.empty
        else pd.DataFrame()
    )

    val_summary = {
        "total_carnival_nodes": int(len(df_node_eval)),
        "nodes_present_in_h5ad": int(len(present_nodes)),
        "nodes_concordant_sign": int(present_nodes["Concordant_Sign"].sum()) if not present_nodes.empty else 0,
        "nodes_concordance_rate": (
            round(float(present_nodes["Concordant_Sign"].mean()), 4) if not present_nodes.empty else 0.0
        ),
        "nodes_significant_fdr": (
            int((present_nodes["FDR"] < fdr_thresh).sum()) if not present_nodes.empty else 0
        ),
        "total_carnival_edges": int(len(df_edge_eval)),
        "edges_present_in_h5ad": int(len(present_edges)),
        "edges_concordant_sign": (
            int(present_edges["Concordant_Interaction"].sum()) if not present_edges.empty else 0
        ),
        "edges_concordance_rate": (
            round(float(present_edges["Concordant_Interaction"].mean()), 4)
            if not present_edges.empty
            else 0.0
        ),
        "edges_significant_fdr": (
            int((present_edges["FDR"] < fdr_thresh).sum()) if not present_edges.empty else 0
        ),
    }

    return df_node_eval, df_edge_eval, val_summary


def build_pathway_network_for_carnival(
    df_nodes: pd.DataFrame,
    df_edges: pd.DataFrame | None,
    net_id: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Constructs a Decoupler network combining:
      1. The global CARNIVAL subnetwork as a signed pathway ('CARNIVAL_Subnetwork_<net_id>'),
      2. Each active CARNIVAL upstream hub with >= tmin_targets ('CARNIVAL_Hub_<Source>'), and
      3. Biological pathways (PROGENy or MSigDB Hallmark) overlapping with CARNIVAL active nodes.
    """
    tmin = int(config["tmin_targets"])
    records: list[dict[str, Any]] = []

    # 1. Global CARNIVAL subnetwork signature
    for row in df_nodes.itertuples(index=False):
        records.append(
            {
                "source": f"CARNIVAL_Subnetwork_{net_id}",
                "target": str(row.Node_ID),
                "weight": float(row.Activity),
            }
        )

    # 2. Local CARNIVAL sub-pathways (respecting Step 04 tmin_targets)
    if df_edges is not None and not df_edges.empty:
        for src, grp in df_edges.groupby("Source"):
            if len(grp) >= tmin:
                for r in grp.itertuples(index=False):
                    records.append(
                        {
                            "source": f"CARNIVAL_Hub_{src}",
                            "target": str(r.Target),
                            "weight": float(r.Interaction) * float(np.sign(r.Activity) or 1.0),
                        }
                    )

    # 3. External canonical pathways filtered by overlap with CARNIVAL nodes
    carnival_node_set = set(df_nodes["Node_ID"].astype(str))
    db_name = str(config["pathway_database"]).lower()
    min_overlap = int(config["min_pathway_carnival_overlap"])

    if db_name != "carnival_only":
        try:
            if db_name == "progeny":
                ext_net = dc.op.progeny(organism="human", top=int(config["progeny_top_genes"]))
            elif db_name == "hallmark":
                ext_net = dc.op.hallmark(organism="human")
            else:
                ext_net = pd.DataFrame()

            if not ext_net.empty:
                src_col = "source" if "source" in ext_net.columns else ext_net.columns[0]
                tgt_col = "target" if "target" in ext_net.columns else ext_net.columns[1]
                w_col = "weight" if "weight" in ext_net.columns else None

                for pw_name, grp in ext_net.groupby(src_col):
                    overlap = set(grp[tgt_col].astype(str)).intersection(carnival_node_set)
                    if len(overlap) >= min_overlap:
                        for r in grp.itertuples(index=False):
                            records.append(
                                {
                                    "source": f"{db_name.upper()}_{pw_name}",
                                    "target": str(getattr(r, tgt_col)),
                                    "weight": float(getattr(r, w_col)) if w_col else 1.0,
                                }
                            )
        except Exception as exc:
            LOGGER.warning("Could not load external pathway resource '%s': %s", db_name, exc)

    net_df = pd.DataFrame(records)
    if not net_df.empty:
        net_df = net_df.drop_duplicates(subset=["source", "target"]).reset_index(drop=True)
    return net_df


def _resolve_methods_for_cohort(
    cohort_label: str,
    cohort_key: str,
    net_meta: dict[str, Any],
    step04_best_combos: dict[str, str],
    config: dict[str, Any],
) -> tuple[list[str], str]:
    """
    Determines which decoupler methods to run for a cohort based on consensus_mode
    and the best_combo recorded in step04_report.json.
    """
    consensus_mode = str(config["consensus_mode"]).lower()
    all_methods: list[str] = config["decoupler_methods"]

    if consensus_mode == "step04_best":
        # Lookup best_combo from Step 04 report by cohort_label, cohort_key, or raw label
        candidates = [
            cohort_label.upper(),
            cohort_key.upper(),
            str(net_meta.get("cohort_raw_label", "")).upper(),
        ]
        for cand in candidates:
            for k04, combo_str in step04_best_combos.items():
                if cand and (cand == k04 or cand.startswith(k04) or k04.startswith(cand)):
                    methods = [m.strip() for m in combo_str.split("+") if m.strip()]
                    if len(methods) >= 2:
                        return methods, combo_str
        LOGGER.warning(
            "No Step 04 best_combo found for '%s'; falling back to full decoupler_methods list.",
            cohort_label,
        )
        return all_methods, " + ".join(all_methods)

    return all_methods, " + ".join(all_methods)


def _compute_stats_from_consensus_acts(
    pw_acts: pd.DataFrame,
    mask_pos: np.ndarray,
    mask_neg: np.ndarray,
    pos_label: str,
    neg_label: str,
) -> pd.DataFrame:
    """Computes High vs Low (or PD vs Control) Welch's t-test, Delta, and BH-FDR on consensus scores."""
    can_test = int(mask_pos.sum()) >= 2 and int(mask_neg.sum()) >= 2
    high_acts = pw_acts.iloc[mask_pos]
    low_acts = pw_acts.iloc[mask_neg]

    diff = high_acts.mean(axis=0) - low_acts.mean(axis=0)

    if can_test:
        t_stat, p_vals = stats.ttest_ind(high_acts, low_acts, equal_var=False, axis=0)
        t_stat = np.nan_to_num(t_stat, nan=0.0)
        p_vals = np.nan_to_num(p_vals, nan=1.0)
        _, padj, _, _ = multipletests(p_vals, method="fdr_bh")
    else:
        t_stat = np.zeros(len(diff), dtype=float)
        p_vals = np.ones(len(diff), dtype=float)
        padj = np.ones(len(diff), dtype=float)

    df_pw_stats = pd.DataFrame(
        {
            "Pathway": diff.index.astype(str),
            f"Mean_Score_{pos_label}": high_acts.mean(axis=0).to_numpy(dtype=float),
            f"Mean_Score_{neg_label}": low_acts.mean(axis=0).to_numpy(dtype=float),
            "Delta_Activity": diff.to_numpy(dtype=float),
            "T_Stat": t_stat,
            "P_Value": p_vals,
            "FDR": padj,
        }
    )
    return df_pw_stats.sort_values("P_Value", ascending=True).reset_index(drop=True)


def score_pathways_on_subjects_step04_aware(
    pdata_norm: ad.AnnData,
    pathway_net: pd.DataFrame,
    group_col: str,
    pos_label: str,
    neg_label: str,
    cohort_label: str,
    cohort_key: str,
    net_meta: dict[str, Any],
    step04_best_combos: dict[str, str],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Scores CARNIVAL subnetworks and biological pathways across subjects using the exact
    Step 04 Decoupler + dc.mt.consensus workflow (inheriting tmin_targets, fdr_threshold,
    diff_threshold, and either the Step 04 best_combo or combo optimization).
    """
    if pathway_net.empty or pdata_norm.n_obs == 0:
        return pd.DataFrame(), pd.DataFrame(), {"status": "empty_network_or_subjects"}

    tmin = int(config["tmin_targets"])
    fdr_thresh = float(config["fdr_threshold"])
    diff_thresh = float(config["diff_threshold"])
    consensus_mode = str(config["consensus_mode"]).lower()

    mask_pos = (pdata_norm.obs[group_col] == pos_label).to_numpy()
    mask_neg = (pdata_norm.obs[group_col] == neg_label).to_numpy()

    methods_to_run, combo_label = _resolve_methods_for_cohort(
        cohort_label=cohort_label,
        cohort_key=cohort_key,
        net_meta=net_meta,
        step04_best_combos=step04_best_combos,
        config=config,
    )

    LOGGER.info(
        "Running dc.mt.decouple on %s (mode=%s, methods=%s, tmin=%d)...",
        cohort_label,
        consensus_mode,
        methods_to_run,
        tmin,
    )

    pdata_run = pdata_norm.copy()
    dc.mt.decouple(
        data=pdata_run,
        net=pathway_net,
        methods=methods_to_run,
        tmin=tmin,
    )
    res_all = {key: pdata_run.obsm[key] for key in pdata_run.obsm.keys()}

    # Mode A: Re-optimize across all combinations (same loop as Step 04)
    if consensus_mode == "step04_optimize":
        min_combo_size = int(config["min_combo_size"])
        combinations: list[tuple[str, ...]] = []
        for r in range(min_combo_size, len(methods_to_run) + 1):
            combinations.extend(itertools.combinations(methods_to_run, r))

        best_acts: pd.DataFrame | None = None
        best_stats: pd.DataFrame | None = None
        best_combo_str: str | None = None
        best_num_sig = -1
        best_max_diff = -1.0

        for combo in combinations:
            res_combo: dict[str, Any] = {}
            valid_m = 0
            for m in combo:
                if f"score_{m}" in res_all:
                    res_combo[f"score_{m}"] = res_all[f"score_{m}"]
                    valid_m += 1
                if f"padj_{m}" in res_all:
                    res_combo[f"padj_{m}"] = res_all[f"padj_{m}"]
            if valid_m < 2:
                continue

            try:
                pw_acts, _ = dc.mt.consensus(res_combo)
                df_stats = _compute_stats_from_consensus_acts(
                    pw_acts, mask_pos, mask_neg, pos_label, neg_label
                )
                sig_mask = (df_stats["FDR"] < fdr_thresh) & (
                    df_stats["Delta_Activity"].abs() > diff_thresh
                )
                n_sig = int(sig_mask.sum())
                max_diff = float(df_stats["Delta_Activity"].abs().max()) if not df_stats.empty else 0.0

                if n_sig > best_num_sig or (n_sig == best_num_sig and max_diff > best_max_diff):
                    best_num_sig = n_sig
                    best_max_diff = max_diff
                    best_acts = pw_acts
                    best_stats = df_stats
                    best_combo_str = " + ".join(combo)
            except Exception as exc:
                LOGGER.debug("Combo %s failed in Step 06: %s", combo, exc)

        if best_acts is None or best_stats is None:
            return pd.DataFrame(), pd.DataFrame(), {"status": "no_valid_consensus_combo"}

        scores_export = best_acts.copy()
        scores_export.insert(0, group_col, pdata_run.obs[group_col].astype(str))
        return scores_export, best_stats, {
            "status": "completed",
            "consensus_mode": consensus_mode,
            "tmin_targets": tmin,
            "combo_used": best_combo_str,
            "fdr_threshold": fdr_thresh,
            "diff_threshold": diff_thresh,
        }

    # Mode B (Default): Use exact best_combo from Step 04 (or custom methods) + dc.mt.consensus
    res_combo = {}
    valid_methods = 0
    for m in methods_to_run:
        if f"score_{m}" in res_all:
            res_combo[f"score_{m}"] = res_all[f"score_{m}"]
            valid_methods += 1
        if f"padj_{m}" in res_all:
            res_combo[f"padj_{m}"] = res_all[f"padj_{m}"]

    if valid_methods >= 2:
        pw_acts, _ = dc.mt.consensus(res_combo)
    elif valid_methods == 1:
        only_score_key = next(k for k in res_combo if k.startswith("score_"))
        pw_acts = pd.DataFrame(res_combo[only_score_key], index=pdata_run.obs_names)
    else:
        return pd.DataFrame(), pd.DataFrame(), {"status": "no_valid_methods_output"}

    df_pw_stats = _compute_stats_from_consensus_acts(
        pw_acts, mask_pos, mask_neg, pos_label, neg_label
    )

    scores_export = pw_acts.copy()
    scores_export.insert(0, group_col, pdata_run.obs[group_col].astype(str))

    return scores_export, df_pw_stats, {
        "status": "completed",
        "consensus_mode": consensus_mode,
        "tmin_targets": tmin,
        "combo_used": combo_label,
        "fdr_threshold": fdr_thresh,
        "diff_threshold": diff_thresh,
    }


def save_step06_plots(
    scores_df: pd.DataFrame,
    df_pw_stats: pd.DataFrame,
    group_col: str,
    pos_label: str,
    neg_label: str,
    net_id: str,
    plots_dir: Path,
    config: dict[str, Any],
) -> dict[str, str]:
    """Generates and saves a pathway delta barplot and a subject-by-pathway activity heatmap."""
    saved_plots: dict[str, str] = {}
    if df_pw_stats.empty or scores_df.empty:
        return saved_plots

    dpi = int(config["dpi_figures"])
    top_n = int(config["top_plot_items"])
    fdr_thresh = float(config["fdr_threshold"])
    diff_thresh = float(config["diff_threshold"])

    # 1. Barplot of Top Differential Pathways across Subjects
    plot_df = df_pw_stats.dropna(subset=["Delta_Activity"]).head(top_n).iloc[::-1]
    if not plot_df.empty:
        fig, ax = plt.subplots(figsize=(9, max(5, len(plot_df) * 0.35)), facecolor="white")
        colors = [
            "#d62728"
            if (r.FDR < fdr_thresh and r.Delta_Activity > diff_thresh)
            else "#1f77b4"
            if (r.FDR < fdr_thresh and r.Delta_Activity < -diff_thresh)
            else "#7f7f7f"
            for r in plot_df.itertuples(index=False)
        ]
        ax.barh(plot_df["Pathway"], plot_df["Delta_Activity"], color=colors, edgecolor="black")
        ax.axvline(0.0, color="black", linewidth=0.9, linestyle="--")
        ax.axvline(diff_thresh, color="grey", linewidth=0.7, linestyle=":")
        ax.axvline(-diff_thresh, color="grey", linewidth=0.7, linestyle=":")
        ax.set_title(f"CARNIVAL & Pathway Consensus Activity ({net_id}: {pos_label} vs {neg_label})")
        ax.set_xlabel(f"Consensus Activity Difference ({pos_label} - {neg_label})")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        barplot_path = build_tagged_path("step06_pathway_activity.png", net_id, default_dir=plots_dir)
        fig.savefig(barplot_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved_plots["pathway_barplot"] = str(barplot_path)

    # 2. Subject-level Heatmap of Pathway Consensus Activities
    numeric_cols = [c for c in scores_df.columns if c != group_col]
    top_pws = [c for c in df_pw_stats["Pathway"].head(top_n) if c in numeric_cols]
    if top_pws:
        ordered_df = scores_df.sort_values(by=group_col)
        mat = ordered_df[top_pws].to_numpy(dtype=float).T

        fig, ax = plt.subplots(
            figsize=(max(8, ordered_df.shape[0] * 0.3), max(5, len(top_pws) * 0.35)),
            facecolor="white",
        )
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r")
        ax.set_yticks(np.arange(len(top_pws)))
        ax.set_yticklabels(top_pws, fontsize=8)
        ax.set_xticks(np.arange(ordered_df.shape[0]))
        ax.set_xticklabels(
            [f"{idx} ({grp})" for idx, grp in zip(ordered_df.index, ordered_df[group_col])],
            rotation=90,
            fontsize=7,
        )
        ax.set_title(f"Subject-Level Consensus Pathway Scores ({net_id})")
        fig.colorbar(im, ax=ax, label="Consensus Activity Score")

        heatmap_path = build_tagged_path("step06_subject_heatmap.png", net_id, default_dir=plots_dir)
        fig.savefig(heatmap_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved_plots["subject_heatmap"] = str(heatmap_path)

    return saved_plots


def format_pipeline_step06_summary(
    *,
    dataset_path: str,
    config_path: str,
    step04_report_path: str,
    networks_evaluated: list[str],
    comparison_mode: str,
    consensus_mode: str,
    tmin_targets: int,
    output_dir: str,
    plots_dir: str,
    report_path: str,
) -> None:
    LOGGER.info(
        "Pipeline Step06 summary:\n"
        f"- Dopaminergic Dataset (.h5ad): {dataset_path}\n"
        f"- Config File: {config_path}\n"
        f"- Step 04 Report (Inherited Params): {step04_report_path}\n"
        f"- CARNIVAL Networks Evaluated ({len(networks_evaluated)}): {', '.join(networks_evaluated)}\n"
        f"- Subject Comparison Mode: {comparison_mode}\n"
        f"- Decoupler Consensus Mode: {consensus_mode} (tmin={tmin_targets})\n"
        f"- Output Directory: {output_dir}\n"
        f"- Plots Directory: {plots_dir}\n"
        f"- Report Path: {report_path}"
    )


def build_step06_parser(default_repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step06 Validation of CARNIVAL Pathways & Subnetworks on .h5ad Subjects (Step04-Aware)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    processed_dir = default_repo_root / "pipeline" / "data" / "processed"
    plots_dir = default_repo_root / "pipeline" / "plots"
    reports_dir = default_repo_root / "pipeline" / "reports"

    parser.add_argument(
        "--config",
        default=str(default_repo_root / "pipeline" / "config" / "step06.yml"),
        help="YAML config file for step06 tunables.",
    )
    parser.add_argument(
        "--dataset-path",
        default=str(processed_dir / CANONICAL_NAMES["dataset_dopaminergic"]),
        help="Path to the Step 02 dopaminergic dataset file (.h5ad).",
    )
    parser.add_argument(
        "--step04-report",
        default=str(reports_dir / "step04_report.json"),
        help="Path to step04_report.json to inherit tmin, stratification, and best_combo methods.",
    )
    parser.add_argument(
        "--step05-report",
        default=str(reports_dir / "step05_report.json"),
        help="Path to step05_report.json for automatic CARNIVAL network discovery.",
    )
    parser.add_argument(
        "--networks-dir",
        default=str(processed_dir / "step05_networks"),
        help="Directory containing Step 05 Carnival_Nodes_*.csv and Carnival_Edges_*.csv files.",
    )
    parser.add_argument(
        "--nodes-csvs",
        nargs="+",
        default=None,
        help="Explicit CARNIVAL nodes CSV files (accepts 'LABEL=path.csv' or 'path.csv').",
    )
    parser.add_argument(
        "--edges-csvs",
        nargs="+",
        default=None,
        help="Optional explicit CARNIVAL edges CSV files matching --nodes-csvs.",
    )
    parser.add_argument(
        "--cohorts",
        nargs="+",
        choices=VALID_COHORTS,
        default=None,
        help="Optional override to evaluate networks on specific subject cohorts in the .h5ad dataset.",
    )
    parser.add_argument(
        "--comparison-mode",
        choices=["proxy", "kiss1r", "condition"],
        default=None,
        help=(
            "Subject grouping for validation (overrides Step 04 / config): 'proxy', 'kiss1r', or 'condition'."
        ),
    )
    parser.add_argument(
        "--consensus-mode",
        choices=["step04_best", "step04_optimize", "custom"],
        default=None,
        help=(
            "How Decoupler consensus is computed: 'step04_best' (uses best_combo from step04_report.json), "
            "'step04_optimize' (searches all method combinations), or 'custom' (uses decoupler_methods)."
        ),
    )
    parser.add_argument(
        "--tmin",
        type=int,
        default=None,
        help="Minimum number of targets per pathway/module in Decoupler (overrides Step 04 / config).",
    )
    parser.add_argument(
        "--pathway-database",
        choices=["progeny", "hallmark", "carnival_only"],
        default=None,
        help="Pathway database to combine with CARNIVAL subnetwork modules (overrides config).",
    )
    parser.add_argument(
        "--min-cells",
        type=int,
        default=None,
        help="Minimum cells per subject in pseudobulk (overrides Step 04 / config).",
    )
    parser.add_argument("--output-dir", default=str(processed_dir / "step06_pathway_validation"))
    parser.add_argument("--plots-dir", default=str(plots_dir))
    parser.add_argument("--reports-dir", default=str(reports_dir))
    parser.add_argument("--logging-level", default="INFO")
    return parser


def run_step06(args: argparse.Namespace) -> None:
    configure_logging(args.logging_level)
    config_path = Path(args.config)
    step04_report_path = Path(args.step04_report)

    config, step04_best_combos = _resolve_step06_config(
        config_path=config_path,
        step04_report_path=step04_report_path,
    )

    if args.min_cells is not None:
        config["min_cells_pseudobulk"] = int(args.min_cells)
    if args.comparison_mode is not None:
        config["comparison_mode"] = str(args.comparison_mode).lower()
    if args.consensus_mode is not None:
        config["consensus_mode"] = str(args.consensus_mode).lower()
    if args.tmin is not None:
        config["tmin_targets"] = int(args.tmin)
    if args.pathway_database is not None:
        config["pathway_database"] = str(args.pathway_database).lower()

    min_cells = int(config["min_cells_pseudobulk"])
    comparison_mode = str(config["comparison_mode"]).lower()
    consensus_mode = str(config["consensus_mode"]).lower()
    tmin_targets = int(config["tmin_targets"])
    fdr_thresh = float(config["fdr_threshold"])
    diff_thresh = float(config["diff_threshold"])
    corr_method = str(config["correlation_method"])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    networks = _discover_step05_networks(
        networks_dir=Path(args.networks_dir),
        step05_report_path=Path(args.step05_report),
        explicit_nodes=args.nodes_csvs,
        explicit_edges=args.edges_csvs,
    )
    if not networks:
        raise FileNotFoundError(
            f"No CARNIVAL network files found in '{args.networks_dir}' or '{args.step05_report}'."
        )

    LOGGER.info("Reading h5ad dataset from %s...", dataset_path)
    adata = sc.read_h5ad(dataset_path)

    pseudobulk_cache: dict[str, tuple[ad.AnnData, str, str, str, dict[str, Any]]] = {}
    report_networks: dict[str, Any] = {}

    for net_id, net_meta in networks.items():
        nodes_csv: Path = net_meta["nodes_csv"]
        edges_csv: Path | None = net_meta["edges_csv"]

        df_nodes = pd.read_csv(nodes_csv)
        df_edges = pd.read_csv(edges_csv) if (edges_csv and edges_csv.exists()) else None

        target_cohorts = args.cohorts if args.cohorts else [net_meta["cohort"]]
        report_networks[net_id] = {
            "nodes_csv": str(nodes_csv),
            "edges_csv": str(edges_csv) if edges_csv else None,
            "evaluations_by_cohort": {},
        }

        for cohort_key in target_cohorts:
            if cohort_key not in pseudobulk_cache:
                cohort_label, sub_adata = _subset_adata_by_cohort(adata, cohort_key)
                pseudobulk_cache[cohort_key] = prepare_subject_pseudobulk(
                    adata=sub_adata,
                    cohort_label=cohort_label,
                    config=config,
                    min_cells=min_cells,
                    comparison_mode=comparison_mode,
                )

            pdata_norm, group_col, pos_label, neg_label, subj_stats = pseudobulk_cache[cohort_key]
            cohort_label = str(subj_stats.get("cohort_label", cohort_key.upper()))
            eval_tag = f"{net_id}_on_{cohort_label}"

            if pdata_norm.n_obs == 0:
                LOGGER.warning("No valid subjects for cohort '%s'. Skipping %s.", cohort_key, eval_tag)
                continue

            LOGGER.info("=== Evaluating CARNIVAL network [%s] on cohort [%s] ===", net_id, cohort_label)

            # 1. Node & Edge validation across subjects
            df_node_eval, df_edge_eval, val_summary = evaluate_carnival_nodes_and_edges_on_subjects(
                pdata_norm=pdata_norm,
                df_nodes=df_nodes,
                df_edges=df_edges,
                group_col=group_col,
                pos_label=pos_label,
                neg_label=neg_label,
                corr_method=corr_method,
                fdr_thresh=fdr_thresh,
            )

            node_eval_csv = build_tagged_path("step06_nodes_validation.csv", eval_tag, default_dir=out_dir)
            df_node_eval.to_csv(node_eval_csv, index=False)

            edge_eval_csv = None
            if not df_edge_eval.empty:
                edge_eval_csv = build_tagged_path("step06_edges_validation.csv", eval_tag, default_dir=out_dir)
                df_edge_eval.to_csv(edge_eval_csv, index=False)

            # 2. Pathway & Subnetwork scoring using Step 04 tmin & consensus methods
            pw_net = build_pathway_network_for_carnival(
                df_nodes=df_nodes,
                df_edges=df_edges,
                net_id=net_id,
                config=config,
            )

            scores_df, df_pw_stats, decoupler_meta = score_pathways_on_subjects_step04_aware(
                pdata_norm=pdata_norm,
                pathway_net=pw_net,
                group_col=group_col,
                pos_label=pos_label,
                neg_label=neg_label,
                cohort_label=cohort_label,
                cohort_key=cohort_key,
                net_meta=net_meta,
                step04_best_combos=step04_best_combos,
                config=config,
            )

            subject_scores_csv = None
            pw_stats_csv = None
            sig_pathways: dict[str, Any] = {}

            if not scores_df.empty and not df_pw_stats.empty:
                subject_scores_csv = build_tagged_path(
                    "step06_subject_pathway_scores.csv", eval_tag, default_dir=out_dir
                )
                scores_df.to_csv(subject_scores_csv)

                pw_stats_csv = build_tagged_path(
                    "step06_pathway_differential_stats.csv", eval_tag, default_dir=out_dir
                )
                df_pw_stats.to_csv(pw_stats_csv, index=False)

                # Filter significant pathways using both Step 04 FDR and diff_threshold
                sig_df = df_pw_stats[
                    (df_pw_stats["FDR"] < fdr_thresh)
                    & (df_pw_stats["Delta_Activity"].abs() > diff_thresh)
                ]
                for row in sig_df.itertuples(index=False):
                    sig_pathways[str(row.Pathway)] = {
                        "delta_activity": round(float(row.Delta_Activity), 6),
                        "t_stat": round(float(row.T_Stat), 4),
                        "p_value": round(float(row.P_Value), 6),
                        "fdr": round(float(row.FDR), 6),
                    }

            # 3. Save Visualizations
            saved_plots = save_step06_plots(
                scores_df=scores_df,
                df_pw_stats=df_pw_stats,
                group_col=group_col,
                pos_label=pos_label,
                neg_label=neg_label,
                net_id=eval_tag,
                plots_dir=plots_dir,
                config=config,
            )

            report_networks[net_id]["evaluations_by_cohort"][cohort_key] = {
                "subjects_summary": subj_stats,
                "decoupler_execution": decoupler_meta,
                "network_validation": val_summary,
                "pathways_tested": int(len(df_pw_stats)),
                "pathways_significant_fdr_and_diff": int(len(sig_pathways)),
                "significant_pathways": sig_pathways,
                "files": {
                    "nodes_validation_csv": str(node_eval_csv),
                    "edges_validation_csv": str(edge_eval_csv) if edge_eval_csv else None,
                    "subject_pathway_scores_csv": str(subject_scores_csv) if subject_scores_csv else None,
                    "pathway_stats_csv": str(pw_stats_csv) if pw_stats_csv else None,
                    "plots": saved_plots,
                },
            }

    report = {
        "config_path": str(config_path),
        "step04_report_path": str(step04_report_path),
        "dataset_path": str(dataset_path),
        "inherited_step04_best_combos": step04_best_combos,
        "parameters": {
            "inherit_step04_params": bool(config["inherit_step04_params"]),
            "consensus_mode": consensus_mode,
            "min_cells_pseudobulk": min_cells,
            "target_sum_norm": float(config["target_sum_norm"]),
            "comparison_mode": comparison_mode,
            "tmin_targets": tmin_targets,
            "fdr_threshold": fdr_thresh,
            "diff_threshold": diff_thresh,
            "pathway_database": config["pathway_database"],
            "correlation_method": corr_method,
        },
        "networks": report_networks,
    }

    report_path = reports_dir / "step06_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    format_pipeline_step06_summary(
        dataset_path=str(dataset_path),
        config_path=str(config_path),
        step04_report_path=str(step04_report_path),
        networks_evaluated=list(networks.keys()),
        comparison_mode=comparison_mode,
        consensus_mode=consensus_mode,
        tmin_targets=tmin_targets,
        output_dir=str(out_dir),
        plots_dir=str(plots_dir),
        report_path=str(report_path),
    )