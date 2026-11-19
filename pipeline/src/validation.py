from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import anndata as ad
import decoupler as dc
import networkx as nx
import numpy as np
import omnipath as op
import pandas as pd
import scanpy as sc
from scipy import stats
from statsmodels.stats.multitest import multipletests

from pipeline.src.schemas import CANONICAL_NAMES
from pipeline.src.tfs import (
    VALID_COHORTS,
    _subset_adata_by_cohort,
    make_pseudobulk_stratified,
)
from pipeline.src.utils import (
    _ensure_list,
    build_tagged_path,
    configure_logging,
    load_yaml_file,
)

LOGGER = logging.getLogger(__name__)

# Canonical stimulus-attributed pathways for KISS1R across KEGG, Reactome, and PROGENy
DEFAULT_KISS1R_ATTRIBUTED_PATHWAYS: list[str] = [
    # --- Pathway KEGG originali ---
    "KEGG_GNRH_SIGNALING_PATHWAY",
    "KEGG_MEDICUS_REFERENCE_KISS1_KISS1R_PLCB_PKC_SIGNALING_PATHWAY",
    "KEGG_MEDICUS_REFERENCE_GNRH_GNRHR_PLCB_PKC_SIGNALING_PATHWAY",
    "KEGG_CALCIUM_SIGNALING_PATHWAY",
    "KEGG_MAPK_SIGNALING_PATHWAY",
    "KEGG_PHOSPHATIDYLINOSITOL_SIGNALING_SYSTEM",
    "KEGG_NEUROTROPHIN_SIGNALING_PATHWAY",
    "KEGG_ERBB_SIGNALING_PATHWAY",
    "KEGG_WNT_SIGNALING_PATHWAY",
    "KEGG_TGF_BETA_SIGNALING_PATHWAY",
    "KEGG_FOCAL_ADHESION",
    "KEGG_VEGF_SIGNALING_PATHWAY",
    "KEGG_LONG_TERM_POTENTIATION",
    "KEGG_VASCULAR_SMOOTH_MUSCLE_CONTRACTION",
    "KEGG_PARKINSONS_DISEASE",
    # --- KEGG aggiuntivi (Autofagia, Dopamina, Stress Ossidativo e Nodi Chiave) ---
    "KEGG_REGULATION_OF_AUTOPHAGY",
    "KEGG_MEDICUS_REFERENCE_PINK_PARKIN_MEDIATED_AUTOPHAGOSOME_FORMATION",
    "KEGG_MEDICUS_REFERENCE_AUTOPHAGY_VESICLE_NUCLEATION_ELONGATION_MATURATION_MTORC1_PI3KC3_C1",
    "KEGG_MEDICUS_REFERENCE_AUTOPHAGY_VESICLE_NUCLEATION_ELONGATION_MATURATION_LC3_II_FORMATION",
    "KEGG_MEDICUS_REFERENCE_DRD1_GNAS_AC_PKA_SIGNALING_PATHWAY",
    "KEGG_MEDICUS_REFERENCE_CHRNA7_E2F_SIGNALING_PATHWAY",
    "KEGG_MEDICUS_REFERENCE_PRNP_PI3K_NOX2_SIGNALING_PATHWAY",
    "KEGG_MEDICUS_REFERENCE_ITGA_B_FAK_RAC_SIGNALING_PATHWAY",
    "KEGG_ENDOCYTOSIS",
    "KEGG_UBIQUITIN_MEDIATED_PROTEOLYSIS",
    "KEGG_AXON_GUIDANCE",
    # --- REACTOME: Autofagia e Mitofagia ---
    "REACTOME_AUTOPHAGY",
    "REACTOME_MACROAUTOPHAGY",
    "REACTOME_SELECTIVE_AUTOPHAGY",
    "REACTOME_CHAPERONE_MEDIATED_AUTOPHAGY",
    "REACTOME_MITOPHAGY",
    "REACTOME_PINK1_PRKN_MEDIATED_MITOPHAGY",
    # --- REACTOME: Segnalazione GPCR / GNAQ / PLCb / PKC / Calcio (Asse KISS1R-GnRH) ---
    "REACTOME_SIGNALING_BY_GPCR",
    "REACTOME_GPCR_DOWNSTREAM_SIGNALLING",
    "REACTOME_G_ALPHA_Q_SIGNALLING_EVENTS",
    "REACTOME_PLC_BETA_MEDIATED_EVENTS",
    "REACTOME_DAG_AND_IP3_SIGNALING",
    "REACTOME_CA2_PATHWAY",
    # --- REACTOME: MAPK / JNK, Recettori Tirosin-Chinasici e Neurotrofine (MAPK8, SRC) ---
    "REACTOME_MAPK_FAMILY_SIGNALING_CASCADES",
    "REACTOME_JNK_C_JUN_KINASES_PHOSPHORYLATION_AND_ACTIVATION_MEDIATED_BY_ACTIVATED_HUMAN_TAK1",
    "REACTOME_SIGNALING_BY_NTRKS",
    "REACTOME_SIGNALING_BY_EGFR",
    "REACTOME_SIGNALING_BY_ERBB2",
    "REACTOME_SIGNALING_BY_VEGF",
    # --- REACTOME: Citoscheletro, Rho GTPasi e Adesione Focale (RAC1, ARHGEF25, SRC, PPP1CA) ---
    "REACTOME_SIGNALING_BY_RHO_GTPASES",
    "REACTOME_RHO_GTPASE_CYCLE",
    "REACTOME_RAC1_GTPASE_CYCLE",
    "REACTOME_RHO_GTPASE_EFFECTORS",
    "REACTOME_NON_INTEGRIN_MEMBRANE_ECM_INTERACTIONS",
    # --- REACTOME: Stress Ossidativo, Apoptosi e Ciclo Cellulare Neuronale (CDK5, RB1, E2F2, CREB1) ---
    "REACTOME_CELLULAR_RESPONSE_TO_STRESS",
    "REACTOME_CELLULAR_RESPONSE_TO_OXIDATIVE_STRESS",
    "REACTOME_FOXO_MEDIATED_TRANSCRIPTION",
    "REACTOME_APOPTOSIS",
    "REACTOME_INTRINSIC_PATHWAY_FOR_APOPTOSIS",
    "REACTOME_G1_S_SPECIFIC_TRANSCRIPTION",
    # --- PROGENy Footprints ---
    "MAPK",
    "EGFR",
    "PI3K",
    "VEGF",
    "TNFa",
    "NFkB",
    "WNT",
    "TGFb",
    "Hypoxia",
    "p53",
    "Trail",
]

# Keywords identifying physiological GPCR/Gq/KISS1R and PD/Autophagy reference cascades across DBs
ATTRIBUTED_PATHWAY_KEYWORDS: tuple[str, ...] = (
    "KISS1",
    "GNRH",
    "PLCB_PKC",
    "G_ALPHA_Q",
    "PLC_BETA",
    "DAG_AND_IP3",
    "CALCIUM_SIGNALING",
    "CA2_PATHWAY",
    "PHOSPHATIDYLINOSITOL_SIGNALING",
    "MAPK_SIGNALING",
    "MAPK_FAMILY",
    "FOCAL_ADHESION",
    "NEUROTROPHIN_SIGNALING",
    "SIGNALING_BY_NTRKS",
    "LONG_TERM_POTENTIATION",
    "AUTOPHAGY",
    "MITOPHAGY",
)

# Supported MSigDB / pathway database patterns (collection substring, pathway prefix)
DATABASE_MATCH_RULES: dict[str, tuple[str, str]] = {
    "kegg": ("kegg", "KEGG_"),
    "reactome": ("reactome", "REACTOME_"),
    "hallmark": ("hallmark", "HALLMARK_"),
    "wikipathways": ("wikipathways", "WP_"),
    "wp": ("wikipathways", "WP_"),
    "biocarta": ("biocarta", "BIOCARTA_"),
    "pid": ("pid", "PID_"),
    "go_bp": ("go_biological_process", "GOBP_"),
    "gobp": ("go_biological_process", "GOBP_"),
    "go_mf": ("go_molecular_function", "GOMF_"),
    "gomf": ("go_molecular_function", "GOMF_"),
    "go_cc": ("go_cellular_component", "GOCC_"),
    "gocc": ("go_cellular_component", "GOCC_"),
}

DEFAULT_STEP06_CONFIG: dict[str, Any] = {
    "min_cells_pseudobulk": 20,
    "target_sum_norm": 10000.0,
    "stratification_mode": "proxy",
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
    # Multi-database support: list of databases (e.g. ["kegg", "reactome", "hallmark", "wikipathways", "go_bp", "progeny"])
    "ora_databases": ["kegg", "reactome"],
    "pathway_gene_sets": "msigdb_kegg_reactome_and_progeny",
    "include_kegg_medicus_reference": True,
    "exclude_kegg_medicus_pathogens_variants": True,
    "include_explicit_attributed_from_msigdb": True,
    "progeny_top_genes": 500,
    "progeny_permutations": 10000,
    "step2_permutations": 10000,
    # Flexible attribution settings: user can pass [], None, or custom names, and toggle defaults/keywords
    "use_default_attributed_pathways": True,
    "attributed_pathways": DEFAULT_KISS1R_ATTRIBUTED_PATHWAYS,
    "use_attributed_keywords": True,
    "attributed_pathway_keywords": list(ATTRIBUTED_PATHWAY_KEYWORDS),
    "n_random_networks": 100,
    "top_n_topology": 20,
    "random_seed": 42,
    "fdr_threshold": 0.05,
}


def _parse_requested_databases(config: dict[str, Any]) -> list[str]:
    """
    Resolves the list of pathway databases requested by the user from either
    `ora_databases` (preferred list/string) or legacy `pathway_gene_sets`.
    """
    raw_dbs = config.get("ora_databases")
    db_tokens: list[str] = []

    if raw_dbs is not None:
        for item in _ensure_list(raw_dbs):
            for part in str(item).replace(",", " ").replace("+", " ").split():
                clean = part.strip().lower()
                if clean:
                    db_tokens.append(clean)

    # Also inspect legacy `pathway_gene_sets` string if `ora_databases` was not explicitly overridden
    mode_str = str(config.get("pathway_gene_sets", "")).lower()
    if not db_tokens and mode_str:
        for db_key in ("kegg", "reactome", "hallmark", "wikipathways", "biocarta", "pid", "go_bp", "gobp"):
            if db_key in mode_str:
                db_tokens.append(db_key)
        if mode_str == "progeny" or (mode_str and not db_tokens and "progeny" in mode_str):
            db_tokens.append("progeny")

    if not db_tokens:
        db_tokens = ["kegg", "reactome"]

    # Normalize aliases while preserving order
    alias_map = {
        "wp": "wikipathways",
        "gobp": "go_bp",
        "gomf": "go_mf",
        "gocc": "go_cc",
    }
    normalized: list[str] = []
    for token in db_tokens:
        norm_t = alias_map.get(token, token)
        if norm_t not in normalized:
            normalized.append(norm_t)
    return normalized


def _resolve_step06_config(
    config_path: Path,
    step04_config_path: Path | None = None,
    cli_ora_databases: list[str] | None = None,
    cli_attributed_pathways: list[str] | None = None,
    cli_disable_keywords: bool = False,
    cli_only_databases: bool = False,
) -> dict[str, Any]:
    config = DEFAULT_STEP06_CONFIG.copy()

    if step04_config_path and step04_config_path.exists():
        loaded04 = load_yaml_file(step04_config_path) or {}
        step04_cfg = loaded04.get("step04", {})
        if isinstance(step04_cfg, dict):
            for shared_key in (
                "min_cells_pseudobulk",
                "target_sum_norm",
                "stratification_mode",
                "kiss1r_gene",
                "gnaq_gene",
                "proxy_genes",
                "min_active_proxies",
            ):
                if shared_key in step04_cfg and step04_cfg[shared_key] is not None:
                    config[shared_key] = step04_cfg[shared_key]

    user_supplied_attributed = False
    if config_path.exists():
        loaded06 = load_yaml_file(config_path) or {}
        step06_cfg = loaded06.get("step06", {})
        if step06_cfg and not isinstance(step06_cfg, dict):
            raise ValueError(f"Expected mapping under 'step06' in config: {config_path}")
        if isinstance(step06_cfg, dict):
            if "attributed_pathways" in step06_cfg:
                user_supplied_attributed = True
            for k, v in step06_cfg.items():
                if v is not None:
                    config[k] = v
                elif k == "attributed_pathways":
                    # Explicit null/empty in YAML means user wants no explicit pathway list
                    config["attributed_pathways"] = []
    else:
        LOGGER.warning(
            "Step06 config file not found (%s), using default/Step04 settings.",
            config_path,
        )

    if cli_ora_databases:
        config["ora_databases"] = cli_ora_databases

    if cli_only_databases:
        config["attributed_pathways"] = []
        config["use_default_attributed_pathways"] = False
        config["use_attributed_keywords"] = False
    elif cli_attributed_pathways is not None:
        config["attributed_pathways"] = cli_attributed_pathways
        user_supplied_attributed = True

    if cli_disable_keywords:
        config["use_attributed_keywords"] = False

    if not bool(config.get("use_default_attributed_pathways", True)) and not user_supplied_attributed:
        config["attributed_pathways"] = []

    config["proxy_genes"] = [str(g) for g in _ensure_list(config["proxy_genes"])]
    config["ora_databases"] = _parse_requested_databases(config)
    config["attributed_pathways"] = [
        str(p).strip()
        for p in _ensure_list(config.get("attributed_pathways", []))
        if str(p).strip()
    ]
    config["attributed_pathway_keywords"] = [
        str(kw).strip()
        for kw in _ensure_list(config.get("attributed_pathway_keywords", []))
        if str(kw).strip()
    ]
    return config


def _reconstruct_omnipath_pkn_universe(
    step05_report: dict[str, Any],
) -> tuple[set[str], dict[str, int]]:
    """
    Reconstructs the exact OmniPath PKN node universe and degree distribution used in Step 05.
    In Liu et al. (2019), all nodes present in the PKN serve as the hypergeometric universe.
    """
    pkn_meta = step05_report.get("omnipath_pkn", {})
    pkn_csv = pkn_meta.get("pkn_nodes_csv")
    if pkn_csv and Path(pkn_csv).exists():
        pkn_df = pd.read_csv(pkn_csv)
        universe = set(pkn_df["Node"].astype(str))
        pkn_degrees = dict(
            zip(pkn_df["Node"].astype(str), pkn_df["all_degree"].astype(int))
        )
        LOGGER.info(
            "Loaded cached OmniPath PKN universe from Step 05: %d nodes.",
            len(universe),
        )
        return universe, pkn_degrees

    organisms = str(pkn_meta.get("organisms", "human"))
    resources = pkn_meta.get(
        "resources",
        [
            "ARN",
            "SIGNOR",
            "SignaLink3",
            "KEGG",
            "ACSN",
            "Reactome_SignaLink3",
            "Reactome_ProtMapper",
        ],
    )
    interaction_types = pkn_meta.get(
        "interaction_types", ["post_translational", "transcriptional"]
    )

    try:
        LOGGER.info("Loading OmniPath PKN universe to match Step 05...")
        knowledge_net = op.interactions.OmniPath.get(
            organisms=organisms,
            resources=resources,
            types=interaction_types,
            genesymbols=True,
        )
        pkn = knowledge_net[knowledge_net["consensus_direction"]].copy()
        pkn["interaction"] = (
            pkn["is_stimulation"].astype(int) - pkn["is_inhibition"].astype(int)
        )
        pkn = pkn[pkn["interaction"] != 0].copy()

        G_pkn = nx.DiGraph()
        for row in pkn.itertuples(index=False):
            src = row.source_genesymbol
            tgt = row.target_genesymbol
            if pd.notna(src) and pd.notna(tgt):
                u_str = str(src).strip()
                v_str = str(tgt).strip()
                if u_str and v_str and u_str != v_str:
                    G_pkn.add_edge(u_str, v_str)

        universe = set(G_pkn.nodes())
        pkn_degrees = dict(G_pkn.degree())
        LOGGER.info("Loaded OmniPath PKN universe: %d unique nodes.", len(universe))
        return universe, pkn_degrees
    except Exception as exc:
        LOGGER.warning(
            "Could not query OmniPath online (%s). Falling back to PROGENy/DoRothEA node union.",
            exc,
        )
        prog_df = dc.op.progeny(top=2000)
        doro_df = dc.op.dorothea(levels=["A", "B", "C"])
        universe = (
            set(prog_df["source"].astype(str))
            .union(set(prog_df["target"].astype(str)))
            .union(set(doro_df["source"].astype(str)))
            .union(set(doro_df["target"].astype(str)))
        )
        return universe, {}


def _infer_pathway_database_label(pw_name: str) -> str:
    pw_up = pw_name.upper()
    for prefix, label in (
        ("KEGG_MEDICUS_", "KEGG_MEDICUS"),
        ("KEGG_", "KEGG"),
        ("REACTOME_", "REACTOME"),
        ("HALLMARK_", "HALLMARK"),
        ("WP_", "WIKIPATHWAYS"),
        ("BIOCARTA_", "BIOCARTA"),
        ("PID_", "PID"),
        ("GOBP_", "GO_BP"),
        ("GOMF_", "GO_MF"),
        ("GOCC_", "GO_CC"),
    ):
        if pw_up.startswith(prefix):
            return label
    return "PROGENY_OR_CUSTOM"


def _load_pathway_gene_sets(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads PROGENy pathway footprints (for multi-granularity transcriptomic validation)
    and curated MSigDB / PROGENy signaling gene sets for Step 1 ORA on CARNIVAL nodes.

    Supports any combination of databases in `config["ora_databases"]`
    (e.g. `["kegg", "reactome", "hallmark", "wikipathways", "biocarta", "pid", "go_bp", "progeny"]`),
    plus optionally pulls any explicitly named pathway in `config["attributed_pathways"]`
    present in MSigDB even if its parent collection was not requested in full.
    """
    top_genes = int(config.get("progeny_top_genes", 500))
    progeny_net = dc.op.progeny(top=top_genes)

    requested_dbs = _parse_requested_databases(config)
    include_medicus_ref = bool(config.get("include_kegg_medicus_reference", True))
    exclude_pathogens_variants = bool(
        config.get("exclude_kegg_medicus_pathogens_variants", True)
    )
    include_explicit_attr = bool(
        config.get("include_explicit_attributed_from_msigdb", True)
    )
    explicit_attr_upper = {
        str(p).strip().upper() for p in config.get("attributed_pathways", []) if str(p).strip()
    }

    ora_frames: list[pd.DataFrame] = []

    msig_dbs = [db for db in requested_dbs if db != "progeny"]
    if msig_dbs or (include_explicit_attr and explicit_attr_upper):
        msig: pd.DataFrame | None = None
        for loader in (
            lambda: dc.op.resource("MSigDB"),
            lambda: dc.op.msigdb(),
        ):
            try:
                msig = loader()
                if msig is not None and not msig.empty:
                    break
            except Exception:
                continue

        if msig is not None and not msig.empty:
            col_set = "collection" if "collection" in msig.columns else None
            gs_col = "geneset" if "geneset" in msig.columns else "source"
            gene_col = "genesymbol" if "genesymbol" in msig.columns else "target"

            gs_series = msig[gs_col].astype(str)
            gs_upper = gs_series.str.upper()
            coll_series = (
                msig[col_set].astype(str).str.lower()
                if col_set
                else pd.Series("", index=msig.index)
            )

            combined_mask = pd.Series(False, index=msig.index)

            for db_key in msig_dbs:
                if db_key in ("all", "msigdb"):
                    combined_mask |= True
                    continue

                if db_key in DATABASE_MATCH_RULES:
                    coll_pat, prefix_pat = DATABASE_MATCH_RULES[db_key]
                    db_mask = gs_upper.str.startswith(prefix_pat)
                    if col_set:
                        db_mask |= coll_series.str.contains(coll_pat, case=False, na=False)
                else:
                    # Generic custom collection or prefix match
                    db_mask = gs_upper.str.startswith(f"{db_key.upper()}_")
                    if col_set:
                        db_mask |= coll_series.str.contains(db_key, case=False, na=False)

                if db_key == "kegg":
                    is_medicus = gs_upper.str.startswith("KEGG_MEDICUS_")
                    if not include_medicus_ref:
                        db_mask &= ~is_medicus
                    elif exclude_pathogens_variants:
                        db_mask &= (~is_medicus) | gs_upper.str.startswith(
                            "KEGG_MEDICUS_REFERENCE_"
                        )

                combined_mask |= db_mask

            # Also include any pathway explicitly listed in `attributed_pathways`
            if include_explicit_attr and explicit_attr_upper:
                combined_mask |= gs_upper.isin(explicit_attr_upper)

            msig_sub = (
                msig.loc[combined_mask, [gs_col, gene_col]]
                .rename(columns={gs_col: "source", gene_col: "target"})
                .dropna()
                .drop_duplicates()
            )
            if not msig_sub.empty:
                ora_frames.append(msig_sub)
        else:
            LOGGER.warning(
                "Could not load MSigDB collections from decoupler; checking fallback sets."
            )

    if "progeny" in requested_dbs or (
        include_explicit_attr
        and explicit_attr_upper.intersection(
            set(progeny_net["source"].astype(str).str.upper())
        )
    ):
        prog_sub = progeny_net[["source", "target"]].dropna().drop_duplicates().copy()
        if "progeny" not in requested_dbs:
            prog_sub = prog_sub[
                prog_sub["source"].astype(str).str.upper().isin(explicit_attr_upper)
            ]
        if not prog_sub.empty:
            ora_frames.append(prog_sub)

    if ora_frames:
        ora_df = pd.concat(ora_frames, ignore_index=True).drop_duplicates(
            subset=["source", "target"]
        )
        db_counts = (
            ora_df[["source"]]
            .drop_duplicates()["source"]
            .astype(str)
            .map(_infer_pathway_database_label)
            .value_counts()
            .to_dict()
        )
        LOGGER.info(
            "Loaded %d total pathways for Two-Step ORA across databases %s (breakdown: %s).",
            ora_df["source"].nunique(),
            requested_dbs,
            db_counts,
        )
        return progeny_net, ora_df

    LOGGER.warning("No MSigDB pathways matched; falling back to PROGENy sets.")
    return progeny_net, progeny_net[["source", "target"]].drop_duplicates()


def _parse_step05_cohort_key(
    step05_cohort_key: str,
    default_strat_mode: str,
) -> tuple[str, str, str]:
    """
    Maps a Step 05 cohort key (e.g., 'PD_MALES_PROXY' or 'PD_MALES') to
    (adata_cohort_key, cohort_label, stratification_mode).
    """
    upper_key = step05_cohort_key.upper()
    strat_mode = default_strat_mode.lower()
    base_label = upper_key

    if upper_key.endswith("_PROXY"):
        strat_mode = "proxy"
        base_label = upper_key[: -len("_PROXY")]
    elif upper_key.endswith("_KISS1R"):
        strat_mode = "kiss1r"
        base_label = upper_key[: -len("_KISS1R")]

    label_to_key = {
        "CONTROL_MALES": "control_male",
        "CONTROL_FEMALES": "control_female",
        "PD_MALES": "pd_male",
        "PD_FEMALES": "pd_female",
        "MALES": "male",
        "FEMALES": "female",
    }
    adata_cohort_key = label_to_key.get(base_label, base_label.lower())
    return adata_cohort_key, base_label, strat_mode


def _load_clean_carnival_tables(
    edges_csv_path: Path,
    nodes_csv_path: Path | None,
    pkn_universe: set[str],
    perturbation_node: str = "KISS1R",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads Step 05 `Carnival_Edges_*.csv` and `Carnival_Nodes_*.csv`,
    filtering out internal Corneto flow vertices ("Unknown", "_*") and preserving
    node biological roles (`Input`, `TF`, `Inferred_Mediator`).
    """
    edges_df = pd.read_csv(edges_csv_path)
    edges_df = edges_df.rename(
        columns={
            "Source": "source",
            "Interaction": "interaction",
            "Target": "target",
            "Activity": "edge_activity",
        }
    )

    edges_df["source"] = edges_df["source"].astype(str).str.strip()
    edges_df["target"] = edges_df["target"].astype(str).str.strip()
    edges_df["edge_activity"] = pd.to_numeric(
        edges_df["edge_activity"], errors="coerce"
    ).fillna(0.0)

    valid_edge_mask = (
        edges_df["source"].notna()
        & edges_df["target"].notna()
        & (edges_df["source"] != "")
        & (edges_df["target"] != "")
        & (edges_df["source"] != "Unknown")
        & (edges_df["target"] != "Unknown")
        & (~edges_df["source"].str.startswith("_"))
        & (~edges_df["target"].str.startswith("_"))
        & (edges_df["edge_activity"].abs() > 0)
    )
    if pkn_universe:
        valid_edge_mask &= edges_df["source"].isin(pkn_universe) & edges_df[
            "target"
        ].isin(pkn_universe)

    edges_df = edges_df.loc[valid_edge_mask].drop_duplicates(
        subset=["source", "interaction", "target"]
    ).copy()

    if nodes_csv_path and nodes_csv_path.exists():
        nodes_df = pd.read_csv(nodes_csv_path)
        nodes_df = nodes_df.rename(columns={"Node_ID": "Node", "Activity": "Activity"})
        nodes_df["Node"] = nodes_df["Node"].astype(str).str.strip()
        nodes_df["Activity"] = pd.to_numeric(
            nodes_df["Activity"], errors="coerce"
        ).fillna(0.0)

        if "Role" not in nodes_df.columns:
            out_deg_zero = set(edges_df["target"]) - set(edges_df["source"])
            nodes_df["Role"] = np.where(
                nodes_df["Node"] == perturbation_node,
                "Input",
                np.where(nodes_df["Node"].isin(out_deg_zero), "TF", "Inferred_Mediator"),
            )

        valid_node_mask = (
            (nodes_df["Node"] != "")
            & (nodes_df["Node"] != "Unknown")
            & (~nodes_df["Node"].str.startswith("_"))
            & (nodes_df["Activity"].abs() > 0)
        )
        if pkn_universe:
            valid_node_mask &= nodes_df["Node"].isin(pkn_universe)

        keep_cols = [
            c
            for c in ("Node", "Activity", "Role", "Input_Measurement")
            if c in nodes_df.columns
        ]
        nodes_df = (
            nodes_df.loc[valid_node_mask, keep_cols]
            .drop_duplicates(subset=["Node"])
            .copy()
        )
    else:
        active_nodes = sorted(
            set(edges_df["source"]).union(set(edges_df["target"]))
        )
        out_deg_zero = set(edges_df["target"]) - set(edges_df["source"])
        roles = [
            "Input"
            if n == perturbation_node
            else ("TF" if n in out_deg_zero else "Inferred_Mediator")
            for n in active_nodes
        ]
        nodes_df = pd.DataFrame(
            {"Node": active_nodes, "Activity": 1.0, "Role": roles}
        )

    return nodes_df, edges_df


def run_two_step_pathway_inference(
    nodes_df: pd.DataFrame,
    pkn_universe: set[str],
    pathway_net: pd.DataFrame,
    attributed_pathways: list[str] | None = None,
    use_attributed_keywords: bool = True,
    attributed_pathway_keywords: list[str] | None = None,
    target_state: int = 1,
    fdr_thresh: float = 0.05,
    n_perm_step2: int = 10000,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Implements the Two-Step Inference Validation from Liu et al. (2019), Fig. 2, Fig. 3 & Methods:
      - Step 1: Hypergeometric Over-Representation Analysis (ORA) on CARNIVAL Up-regulated
        and Down-regulated nodes across all user-selected pathway databases (KEGG, Reactome, etc.)
        using all PKN nodes as the background universe.
      - Step 2: Pathway-Set Enrichment Analysis (EA) on Step 1 results using the 4 statistics
        from `piano::runGSA` (Stouffer, Tail-Strength, Wilcoxon, Reporter) calibrated with
        10,000 permutations across the pathway universe. If the user provides only pathway
        databases without attributed pathway names or keywords, Step 1 ORA completes normally
        and Step 2 is cleanly marked as skipped.
    """
    universe = set(pkn_universe).union(set(nodes_df["Node"].astype(str)))
    M_universe = max(len(universe), 1)

    nodes_up = set(
        nodes_df.loc[nodes_df["Activity"] > 0, "Node"].astype(str)
    ).intersection(universe)
    nodes_down = set(
        nodes_df.loc[nodes_df["Activity"] < 0, "Node"].astype(str)
    ).intersection(universe)

    role_map = (
        dict(zip(nodes_df["Node"].astype(str), nodes_df["Role"].astype(str)))
        if "Role" in nodes_df.columns
        else {}
    )

    attr_set_upper = {str(p).strip().upper() for p in (attributed_pathways or []) if str(p).strip()}
    kw_tuple_upper = tuple(
        str(kw).strip().upper()
        for kw in (
            attributed_pathway_keywords
            if attributed_pathway_keywords is not None
            else ATTRIBUTED_PATHWAY_KEYWORDS
        )
        if str(kw).strip()
    )

    def _is_attributed(pw_name: str) -> bool:
        pw_up = pw_name.upper()
        if pw_up in attr_set_upper:
            return True
        if use_attributed_keywords and kw_tuple_upper:
            return any(kw in pw_up for kw in kw_tuple_upper)
        return False

    pw_groups: dict[str, set[str]] = {}
    for pw_name, grp in pathway_net.groupby("source"):
        members = set(grp["target"].astype(str)).intersection(universe)
        if len(members) >= 3:
            pw_groups[str(pw_name)] = members

    def _step1_ora(test_nodes: set[str], direction_label: str) -> pd.DataFrame:
        N_drawn = len(test_nodes)
        rows = []
        for pw_name, pw_members in pw_groups.items():
            K_pw = len(pw_members)
            overlap_genes = sorted(test_nodes.intersection(pw_members))
            k_overlap = len(overlap_genes)

            mediators_in_overlap = [
                g for g in overlap_genes if role_map.get(g) == "Inferred_Mediator"
            ]
            tfs_in_overlap = [g for g in overlap_genes if role_map.get(g) == "TF"]

            if k_overlap >= 1 and N_drawn >= 1:
                pval = float(stats.hypergeom.sf(k_overlap - 1, M_universe, K_pw, N_drawn))
            else:
                pval = np.nan

            rows.append(
                {
                    "Pathway": pw_name,
                    "Database": _infer_pathway_database_label(pw_name),
                    "Direction": direction_label,
                    "Overlap_Count": k_overlap,
                    "Inferred_Mediator_Count": len(mediators_in_overlap),
                    "TF_Count": len(tfs_in_overlap),
                    "Pathway_Size_in_PKN": K_pw,
                    "CARNIVAL_Set_Size": N_drawn,
                    "PKN_Universe_Size": M_universe,
                    "pval": pval,
                    "Is_Attributed": _is_attributed(pw_name),
                    "Overlap_Nodes": ";".join(overlap_genes),
                    "Mediator_Overlap_Nodes": ";".join(mediators_in_overlap),
                }
            )

        df_ora = pd.DataFrame(rows)
        if df_ora.empty:
            return df_ora

        tested_mask = df_ora["pval"].notna()
        padj_arr = np.ones(len(df_ora), dtype=float)
        if tested_mask.any():
            _, padj_tested, _, _ = multipletests(
                df_ora.loc[tested_mask, "pval"].to_numpy(dtype=float),
                method="fdr_bh",
            )
            padj_arr[tested_mask.to_numpy()] = padj_tested

        df_ora["tested_in_ora"] = tested_mask
        df_ora["pval"] = df_ora["pval"].fillna(1.0)
        df_ora["padj"] = padj_arr
        df_ora["minus_log10_padj"] = -np.log10(np.clip(df_ora["padj"], 1e-16, 1.0))
        return df_ora.sort_values(
            ["padj", "pval", "Overlap_Count"], ascending=[True, True, False]
        )

    ora_up_df = _step1_ora(nodes_up, "Up")
    ora_down_df = _step1_ora(nodes_down, "Down")
    ora_combined_df = pd.concat([ora_up_df, ora_down_df], ignore_index=True)

    def _step2_attributed_ea(ora_df: pd.DataFrame, rng_seed: int) -> dict[str, Any]:
        """
        Executes Step 2 Enrichment Analysis on Step 1 FDR-adjusted p-values using
        the 4 methods from Liu et al. (2019) (stouffer, tailStrength, wilcoxon, reporter)
        with 10,000 permutations over the pathway collection.
        """
        if ora_df.empty:
            return {
                "status": "empty_ora",
                "n_attributed_pathways_in_collection": 0,
                "n_attributed_pathways_active": 0,
                "mean_pval": 1.0,
                "std_pval": 0.0,
                "median_minus_log10_pval": 0.0,
                "iqr_minus_log10_pval": 0.0,
                "significant": False,
                "methods": {},
            }

        attr_mask = ora_df["Is_Attributed"].to_numpy(dtype=bool)
        n_total_pw = len(ora_df)
        m_attr = int(attr_mask.sum())

        if m_attr == 0 or m_attr >= n_total_pw:
            return {
                "status": (
                    "skipped_no_attributed_pathways"
                    if m_attr == 0
                    else "skipped_all_pathways_attributed"
                ),
                "n_attributed_pathways_in_collection": m_attr,
                "n_attributed_pathways_active": 0,
                "mean_pval": 1.0,
                "std_pval": 0.0,
                "median_minus_log10_pval": 0.0,
                "iqr_minus_log10_pval": 0.0,
                "significant": False,
                "methods": {},
            }

        padj_all = np.clip(ora_df["padj"].to_numpy(dtype=float), 1e-12, 1.0)
        neg_log10_all = -np.log10(padj_all)

        p_for_z = np.where(padj_all >= 1.0, 0.5, np.clip(padj_all, 1e-12, 0.9999))
        z_all = stats.norm.isf(p_for_z)

        rng = np.random.default_rng(rng_seed)
        n_perm = min(max(n_perm_step2, 1000), 20000)
        rand_indices = np.argsort(rng.random((n_perm, n_total_pw)), axis=1)[:, :m_attr]

        # 1. Stouffer's method
        obs_stouffer = float(np.sum(z_all[attr_mask]) / np.sqrt(m_attr))
        null_stouffer = np.sum(z_all[rand_indices], axis=1) / np.sqrt(m_attr)
        p_stouffer = float(
            (np.sum(null_stouffer >= obs_stouffer) + 1.0) / (n_perm + 1.0)
        )

        # 2. Tail-Strength method
        expected_u = np.arange(1, m_attr + 1) / (m_attr + 1.0)
        obs_p_sorted = np.sort(padj_all[attr_mask])
        obs_ts = float(np.mean(1.0 - obs_p_sorted / expected_u))
        null_p_sorted = np.sort(padj_all[rand_indices], axis=1)
        null_ts = np.mean(1.0 - null_p_sorted / expected_u[None, :], axis=1)
        p_tail = float((np.sum(null_ts >= obs_ts) + 1.0) / (n_perm + 1.0))

        # 3. Wilcoxon rank-sum test
        if np.any(neg_log10_all > 0):
            _, p_wilcox = stats.mannwhitneyu(
                neg_log10_all[attr_mask],
                neg_log10_all[~attr_mask],
                alternative="greater",
            )
            p_wilcox = float(p_wilcox)
        else:
            p_wilcox = 1.0

        # 4. Reporter method
        obs_reporter = float(np.mean(neg_log10_all[attr_mask]))
        null_reporter = np.mean(neg_log10_all[rand_indices], axis=1)
        mu_rep = float(np.mean(null_reporter))
        sd_rep = float(np.std(null_reporter))
        if sd_rep > 1e-12:
            z_rep = (obs_reporter - mu_rep) / sd_rep
            p_reporter = float(stats.norm.sf(z_rep))
        else:
            p_reporter = 1.0

        methods_dict = {
            "stouffer": round(p_stouffer, 6),
            "tailStrength": round(p_tail, 6),
            "wilcoxon": round(p_wilcox, 6),
            "reporter": round(p_reporter, 6),
        }
        p_vec = np.array(list(methods_dict.values()), dtype=float)
        neg_log10_vec = -np.log10(np.clip(p_vec, 1e-12, 1.0))

        mean_p = float(np.mean(p_vec))
        std_p = float(np.std(p_vec))
        med_log10 = float(np.median(neg_log10_vec))
        iqr_log10 = float(
            np.percentile(neg_log10_vec, 75) - np.percentile(neg_log10_vec, 25)
        )

        is_sig = bool(mean_p < fdr_thresh or float(np.median(p_vec)) < fdr_thresh)

        return {
            "status": "completed",
            "n_attributed_pathways_in_collection": m_attr,
            "n_attributed_pathways_active": int(
                (ora_df.loc[attr_mask, "Overlap_Count"] >= 1).sum()
            ),
            "mean_pval": round(mean_p, 6),
            "std_pval": round(std_p, 6),
            "median_minus_log10_pval": round(med_log10, 4),
            "iqr_minus_log10_pval": round(iqr_log10, 4),
            "significant": is_sig,
            "methods": methods_dict,
        }

    ea_up = _step2_attributed_ea(ora_up_df, seed)
    ea_down = _step2_attributed_ea(ora_down_df, seed + 1)

    if target_state >= 0:
        clear_directionality = bool(ea_up["significant"] and not ea_down["significant"])
    else:
        clear_directionality = bool(ea_down["significant"] and not ea_up["significant"])

    db_breakdown = (
        pd.Series([_infer_pathway_database_label(k) for k in pw_groups.keys()])
        .value_counts()
        .to_dict()
        if pw_groups
        else {}
    )

    summary = {
        "target_state": target_state,
        "n_nodes_up": len(nodes_up),
        "n_nodes_down": len(nodes_down),
        "pkn_universe_size": M_universe,
        "total_pathways_in_collection": len(pw_groups),
        "total_kegg_pathways_in_collection": int(
            db_breakdown.get("KEGG", 0) + db_breakdown.get("KEGG_MEDICUS", 0)
        ),
        "pathways_in_collection_by_database": db_breakdown,
        "attribution_mode": {
            "explicit_attributed_count": len(attr_set_upper),
            "use_attributed_keywords": bool(use_attributed_keywords),
            "keywords_count": len(kw_tuple_upper) if use_attributed_keywords else 0,
        },
        "step1_top_up_pathways": ora_up_df[ora_up_df["Overlap_Count"] >= 1]
        .head(10)
        .to_dict(orient="records"),
        "step1_top_down_pathways": ora_down_df[ora_down_df["Overlap_Count"] >= 1]
        .head(10)
        .to_dict(orient="records"),
        "step2_attributed_enrichment_up": ea_up,
        "step2_attributed_enrichment_down": ea_down,
        "clear_directionality": clear_directionality,
    }
    return ora_combined_df, summary


def run_topological_and_randomization_analysis(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    pkn_degrees: dict[str, int],
    n_random: int = 100,
    top_n: int = 20,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Implements the Network Topological Analysis from Liu et al. (2019), Results, Methods & Fig. S4.
    """
    G = nx.DiGraph()
    node_act_map = dict(
        zip(nodes_df["Node"].astype(str), nodes_df["Activity"].astype(float))
    )
    node_role_map = (
        dict(zip(nodes_df["Node"].astype(str), nodes_df["Role"].astype(str)))
        if "Role" in nodes_df.columns
        else {}
    )

    for row in edges_df.itertuples(index=False):
        u = str(row.source)
        v = str(row.target)
        act = float(row.edge_activity)
        sign = int(getattr(row, "interaction", np.sign(act)))
        G.add_node(u, activity=node_act_map.get(u, 1.0))
        G.add_node(v, activity=node_act_map.get(v, 1.0))
        G.add_edge(u, v, weight=max(abs(act), 1e-6), sign=sign)

    if G.number_of_nodes() == 0:
        return pd.DataFrame(), {"status": "empty_network"}

    def _extract_metrics(graph: nx.DiGraph) -> dict[str, dict[str, float]]:
        in_d = dict(graph.in_degree())
        out_d = dict(graph.out_degree())
        all_d = dict(graph.degree())
        betw = nx.betweenness_centrality(graph, normalized=True)
        try:
            hubs, auths = nx.hits(graph, max_iter=500, tol=1e-6, normalized=True)
        except Exception:
            n_v = max(graph.number_of_nodes(), 1)
            hubs = {n: out_d[n] / n_v for n in graph.nodes()}
            auths = {n: in_d[n] / n_v for n in graph.nodes()}
        return {
            "in_degree": in_d,
            "out_degree": out_d,
            "all_degree": all_d,
            "betweenness": betw,
            "hub_score": hubs,
            "authority_score": auths,
        }

    obs_m = _extract_metrics(G)

    rng = np.random.default_rng(seed)
    rand_betw: dict[str, list[float]] = {n: [] for n in G.nodes()}
    rand_hub: dict[str, list[float]] = {n: [] for n in G.nodes()}
    rand_auth: dict[str, list[float]] = {n: [] for n in G.nodes()}

    nodes_list = list(G.nodes())
    n_nodes = len(nodes_list)
    pos_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("sign", 1) >= 0]
    neg_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("sign", 1) < 0]

    for _ in range(n_random):
        G_rand = nx.DiGraph()
        G_rand.add_nodes_from(nodes_list)
        perm_map = dict(zip(nodes_list, rng.permutation(nodes_list)))

        for edge_subset, sign_val in ((pos_edges, 1), (neg_edges, -1)):
            G_sub = nx.DiGraph()
            G_sub.add_edges_from(edge_subset)
            if G_sub.number_of_edges() >= 4:
                try:
                    nx.directed_edge_swap(
                        G_sub,
                        nswap=max(G_sub.number_of_edges() * 2, 8),
                        max_tries=max(G_sub.number_of_edges() * 20, 100),
                        seed=int(rng.integers(0, 1_000_000)),
                    )
                except Exception:
                    pass
            for u, v in G_sub.edges():
                G_rand.add_edge(perm_map[u], perm_map[v], sign=sign_val, weight=1.0)

        rm = _extract_metrics(G_rand)
        for n in nodes_list:
            act_w = max(abs(float(node_act_map.get(n, 1.0))), 1e-3)
            rand_betw[n].append(float(rm["betweenness"].get(n, 0.0)) * act_w)
            rand_hub[n].append(float(rm["hub_score"].get(n, 0.0)) * act_w)
            rand_auth[n].append(float(rm["authority_score"].get(n, 0.0)) * act_w)

    all_pkn_deg_vals = (
        np.array(list(pkn_degrees.values()), dtype=float)
        if pkn_degrees
        else np.array([1.0])
    )

    rows = []
    for n in nodes_list:
        act_val = float(node_act_map.get(n, 1.0))
        act_w = max(abs(act_val), 1e-3)
        role_str = node_role_map.get(n, "Inferred_Mediator")

        b_val = float(obs_m["betweenness"][n]) * act_w
        h_val = float(obs_m["hub_score"][n]) * act_w
        a_val = float(obs_m["authority_score"][n]) * act_w

        b_arr = np.array(rand_betw[n], dtype=float)
        h_arr = np.array(rand_hub[n], dtype=float)
        a_arr = np.array(rand_auth[n], dtype=float)

        def _pval_vs_rand(obs_v: float, r_arr: np.ndarray) -> float:
            if len(r_arr) >= 5 and np.std(r_arr) > 1e-12:
                _, pv = stats.ttest_1samp(r_arr, popmean=obs_v, alternative="less")
                return float(pv)
            return 1.0

        pkn_deg = int(pkn_degrees.get(n, 0))
        pkn_pct = float(np.mean(all_pkn_deg_vals <= pkn_deg))

        rows.append(
            {
                "Node": n,
                "Role": role_str,
                "Activity": round(act_val, 4),
                "in_degree": int(obs_m["in_degree"][n]),
                "out_degree": int(obs_m["out_degree"][n]),
                "all_degree": int(obs_m["all_degree"][n]),
                "pkn_degree": pkn_deg,
                "pkn_degree_percentile": round(pkn_pct, 4),
                "betweenness": round(b_val, 6),
                "hub_score": round(h_val, 6),
                "authority_score": round(a_val, 6),
                "rand_betweenness_mean": round(
                    float(np.mean(b_arr)) if len(b_arr) else 0.0, 6
                ),
                "rand_hub_mean": round(
                    float(np.mean(h_arr)) if len(h_arr) else 0.0, 6
                ),
                "rand_authority_mean": round(
                    float(np.mean(a_arr)) if len(a_arr) else 0.0, 6
                ),
                "betweenness_vs_rand_pval": round(_pval_vs_rand(b_val, b_arr), 6),
                "hub_vs_rand_pval": round(_pval_vs_rand(h_val, h_arr), 6),
                "authority_vs_rand_pval": round(_pval_vs_rand(a_val, a_arr), 6),
            }
        )

    topo_df = pd.DataFrame(rows).sort_values(
        by=["all_degree", "betweenness", "hub_score"],
        ascending=[False, False, False],
    )

    mediators_df = topo_df[topo_df["Role"] == "Inferred_Mediator"]
    if len(mediators_df) >= 2:
        _, med_topo_pval = stats.ttest_ind(
            mediators_df["betweenness"].to_numpy(dtype=float),
            mediators_df["rand_betweenness_mean"].to_numpy(dtype=float),
            equal_var=False,
        )
        med_topo_pval = float(np.nan_to_num(med_topo_pval, nan=1.0))
    else:
        med_topo_pval = 1.0

    carnival_pkn_degs = topo_df["pkn_degree"].to_numpy(dtype=float)
    non_hub_mediators = (
        mediators_df[mediators_df["pkn_degree"] < np.percentile(all_pkn_deg_vals, 90)][
            "Node"
        ].tolist()
        if not mediators_df.empty
        else []
    )

    top_categories = {
        metric: topo_df.sort_values(metric, ascending=False)
        .head(top_n)[["Node", "Role", "Activity", metric, "pkn_degree", "betweenness_vs_rand_pval"]]
        .to_dict(orient="records")
        for metric in (
            "all_degree",
            "in_degree",
            "out_degree",
            "betweenness",
            "hub_score",
            "authority_score",
        )
    }

    summary = {
        "n_active_nodes": n_nodes,
        "n_active_edges": int(G.number_of_edges()),
        "n_inferred_mediators": int((topo_df["Role"] == "Inferred_Mediator").sum()),
        "n_explained_tfs": int((topo_df["Role"] == "TF").sum()),
        "n_randomized_networks": n_random,
        "mediators_betweenness_vs_random_pval": round(med_topo_pval, 6),
        "median_carnival_node_pkn_degree": round(float(np.median(carnival_pkn_degs)), 2),
        "median_full_pkn_degree": round(float(np.median(all_pkn_deg_vals)), 2),
        "context_specific_non_hub_mediators": non_hub_mediators,
        "top_nodes_by_category": top_categories,
    }
    return topo_df, summary


def compare_opposite_target_states(
    state_networks: dict[int, tuple[pd.DataFrame, pd.DataFrame]],
    perturbation_node: str = "KISS1R",
    canonical_effector: str = "GNAQ",
) -> dict[str, Any]:
    """
    Computes Jaccard similarity index (|A n B| / |A u B|), sign inversion rate,
    and parsimony/mechanistic comparison between KISS1R=+1 and KISS1R=-1 CARNIVAL networks.
    """
    if 1 not in state_networks or -1 not in state_networks:
        return {"status": "single_state_only"}

    nodes_pos, edges_pos = state_networks[1]
    nodes_neg, edges_neg = state_networks[-1]

    set_n_pos = set(nodes_pos["Node"].astype(str))
    set_n_neg = set(nodes_neg["Node"].astype(str))
    node_union = set_n_pos.union(set_n_neg)
    node_inter = set_n_pos.intersection(set_n_neg)
    node_jaccard = len(node_inter) / max(len(node_union), 1)

    set_e_pos = set(
        zip(edges_pos["source"].astype(str), edges_pos["target"].astype(str))
    )
    set_e_neg = set(
        zip(edges_neg["source"].astype(str), edges_neg["target"].astype(str))
    )
    edge_union = set_e_pos.union(set_e_neg)
    edge_inter = set_e_pos.intersection(set_e_neg)
    edge_jaccard = len(edge_inter) / max(len(edge_union), 1)

    map_pos = dict(
        zip(nodes_pos["Node"].astype(str), nodes_pos["Activity"].astype(float))
    )
    map_neg = dict(
        zip(nodes_neg["Node"].astype(str), nodes_neg["Activity"].astype(float))
    )
    flipped_nodes = sorted(
        [
            n
            for n in node_inter
            if np.sign(map_pos.get(n, 0.0)) == -np.sign(map_neg.get(n, 0.0))
        ]
    )

    more_parsimonious_state = (
        1 if len(edges_pos) <= len(edges_neg) else -1
    )
    canonical_gq_active_in_state_1 = bool(
        map_pos.get(perturbation_node, 0.0) > 0
        and map_pos.get(canonical_effector, 0.0) > 0
    )

    return {
        "status": "completed",
        "state_1_nodes_count": len(set_n_pos),
        "state_minus1_nodes_count": len(set_n_neg),
        "shared_nodes_count": len(node_inter),
        "node_jaccard_similarity": round(float(node_jaccard), 4),
        "state_1_edges_count": len(set_e_pos),
        "state_minus1_edges_count": len(set_e_neg),
        "shared_edges_count": len(edge_inter),
        "edge_jaccard_similarity": round(float(edge_jaccard), 4),
        "sign_inverted_shared_nodes_freq": round(
            len(flipped_nodes) / max(len(node_inter), 1), 4
        ),
        "sign_inverted_nodes": flipped_nodes,
        "more_parsimonious_state": more_parsimonious_state,
        "canonical_kiss1r_gnaq_activation_in_state_1": canonical_gq_active_in_state_1,
    }


def run_progeny_footprint_validation(
    pdata_norm: ad.AnnData,
    progeny_net: pd.DataFrame,
    n_perm: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Computes PROGENy 14 signaling pathway activities from pseudobulk expression.
    """
    dc.mt.ulm(data=pdata_norm, net=progeny_net, tmin=5)
    pw_acts = pdata_norm.obsm["score_ulm"]

    mask_high = (pdata_norm.obs["PROXY_status"] == "High").to_numpy()
    mask_low = (pdata_norm.obs["PROXY_status"] == "Low").to_numpy()

    high_acts = pw_acts.iloc[mask_high]
    low_acts = pw_acts.iloc[mask_low]

    diff = high_acts.mean(axis=0) - low_acts.mean(axis=0)
    t_stat, p_vals = stats.ttest_ind(high_acts, low_acts, equal_var=False, axis=0)
    t_stat = np.nan_to_num(t_stat, nan=0.0)
    p_vals = np.nan_to_num(p_vals, nan=1.0)
    _, padj, _, _ = multipletests(p_vals, method="fdr_bh")

    X_mat = (
        pdata_norm.X.toarray()
        if hasattr(pdata_norm.X, "toarray")
        else np.asarray(pdata_norm.X, dtype=float)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        gene_t_vals, _ = stats.ttest_ind(
            X_mat[mask_high, :], X_mat[mask_low, :], equal_var=False, axis=0
        )
    gene_t_vals = np.nan_to_num(gene_t_vals, nan=0.0)
    gene_index_map = {str(g): idx for idx, g in enumerate(pdata_norm.var_names)}

    weight_col = "weight" if "weight" in progeny_net.columns else None
    rng = np.random.default_rng(seed)
    n_perm_eff = min(max(n_perm, 1000), 10000)
    n_genes_total = len(gene_t_vals)

    eq1_scores: list[float] = []
    emp_pvals: list[float] = []

    for pw_name in diff.index:
        sub_pw = progeny_net[progeny_net["source"].astype(str) == str(pw_name)]
        valid_rows = [
            (
                gene_index_map[str(r.target)],
                float(getattr(r, weight_col, 1.0)) if weight_col else 1.0,
            )
            for r in sub_pw.itertuples(index=False)
            if str(r.target) in gene_index_map
        ]
        if not valid_rows:
            eq1_scores.append(0.0)
            emp_pvals.append(1.0)
            continue

        g_idx = np.array([x[0] for x in valid_rows], dtype=int)
        w_vec = np.array([x[1] for x in valid_rows], dtype=float)
        obs_footprint = float(np.dot(gene_t_vals[g_idx], w_vec))

        rand_g_idx = rng.integers(0, n_genes_total, size=(n_perm_eff, len(g_idx)))
        null_footprints = gene_t_vals[rand_g_idx] @ w_vec

        pct = float(np.mean(null_footprints <= obs_footprint))
        eq1_scores.append(2.0 * (pct - 0.5))
        emp_p = float(
            (np.sum(np.abs(null_footprints) >= abs(obs_footprint)) + 1.0)
            / (n_perm_eff + 1.0)
        )
        emp_pvals.append(emp_p)

    return pd.DataFrame(
        {
            "Pathway": diff.index,
            "progeny_delta_activity": np.round(diff.to_numpy(dtype=float), 6),
            "progeny_t_value": np.round(t_stat, 6),
            "progeny_eq1_score": np.round(eq1_scores, 6),
            "progeny_gene_perm_pval": np.round(emp_pvals, 6),
            "pval": np.round(p_vals, 6),
            "padj": np.round(padj, 6),
        }
    ).sort_values("pval")


def integrate_step04_tf_stability(
    nodes_df: pd.DataFrame,
    step04_dir: Path,
    cohort_label: str,
    strat_mode: str,
) -> dict[str, Any]:
    """
    Integrates the Step 04 `TF_Stability_*.csv` audit table with the active TFs
    in the CARNIVAL network.
    """
    strat_tag = strat_mode.upper()
    candidates = [
        build_tagged_path(
            Path("TF_Stability.csv"), cohort_label, strat_tag, default_dir=step04_dir
        ),
        step04_dir / f"TF_Stability_{cohort_label}_{strat_tag}.csv",
        step04_dir / f"TF_Stability_{cohort_label}.csv",
    ]
    stab_path = next((p for p in candidates if p.exists()), None)
    if stab_path is None:
        LOGGER.warning(
            "Step 04 TF stability CSV not found for %s (%s) in %s.",
            cohort_label,
            strat_tag,
            step04_dir,
        )
        return {"status": "missing_stability_csv", "retained_tfs": []}

    stab_df = pd.read_csv(stab_path)
    node_act_map = dict(
        zip(nodes_df["Node"].astype(str), nodes_df["Activity"].astype(float))
    )

    retained = stab_df[stab_df["Node"].astype(str).isin(node_act_map.keys())].copy()
    if retained.empty:
        return {
            "status": "no_tf_overlap_in_network",
            "stability_csv": str(stab_path),
            "retained_tfs": [],
        }

    retained["carnival_activity"] = retained["Node"].astype(str).map(node_act_map)
    retained["matches_primary_sign"] = np.sign(
        retained["carnival_activity"]
    ) == np.sign(retained["primary_delta_activity"])

    cols_to_export = [
        c
        for c in (
            "Node",
            "carnival_activity",
            "primary_delta_activity",
            "primary_pval",
            "primary_padj",
            "matches_primary_sign",
            "single_method_sign_concordance",
            "single_methods_nominal_sig",
            "combo_fdr_selection_freq",
            "combo_nominal_selection_freq",
            "combo_median_padj",
            "config_sign_concordance_freq",
            "config_nominal_stability_freq",
            "kiss1r_pure_delta",
            "kiss1r_pure_pval",
        )
        if c in retained.columns
    ]

    return {
        "status": "completed",
        "stability_csv": str(stab_path),
        "n_network_tfs_audited": int(len(retained)),
        "sign_match_rate_with_step04": round(
            float(retained["matches_primary_sign"].mean()), 4
        ),
        "mean_single_method_sign_concordance": round(
            float(retained["single_method_sign_concordance"].mean()), 4
        ),
        "mean_combo_fdr_selection_freq": round(
            float(retained["combo_fdr_selection_freq"].mean()), 4
        ),
        "mean_combo_nominal_selection_freq": round(
            float(retained["combo_nominal_selection_freq"].mean()), 4
        )
        if "combo_nominal_selection_freq" in retained.columns
        else 0.0,
        "mean_config_sign_concordance_freq": round(
            float(retained["config_sign_concordance_freq"].mean()), 4
        ),
        "all_retained_tfs_nominally_sig_in_pure_kiss1r": bool(
            (retained["kiss1r_pure_pval"] < 0.05).all()
        )
        if "kiss1r_pure_pval" in retained.columns
        else False,
        "retained_tfs": retained[cols_to_export].to_dict(orient="records"),
    }


def build_step06_parser(default_repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Step06 CARNIVAL Network Validation (Multi-Database Two-Step ORA/EA, "
            "Topology vs 100 Randomized Networks, PROGENy Footprints & Step04 TF Stability)"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    processed_dir = default_repo_root / "pipeline" / "data" / "processed"
    reports_dir = default_repo_root / "pipeline" / "reports"

    parser.add_argument(
        "--config",
        default=str(default_repo_root / "pipeline" / "config" / "step06.yml"),
        help="YAML config file for step06 tunables.",
    )
    parser.add_argument(
        "--step04-config",
        default=str(default_repo_root / "pipeline" / "config" / "step04.yml"),
        help="YAML config file for step04 to inherit stratification settings.",
    )
    parser.add_argument(
        "--dataset-path",
        default=str(processed_dir / CANONICAL_NAMES["dataset_dopaminergic"]),
        help="Path to the dopaminergic dataset file (.h5ad).",
    )
    parser.add_argument(
        "--step04-dir",
        default=str(processed_dir / "step04_carnival"),
        help="Directory containing Step 04 TF measurements and stability CSVs.",
    )
    parser.add_argument(
        "--step05-dir",
        default=str(processed_dir / "step05_networks"),
        help="Directory containing Step 05 Carnival_Edges and Carnival_Nodes CSVs.",
    )
    parser.add_argument(
        "--step05-report",
        default=str(reports_dir / "step05_report.json"),
        help="Path to step05_report.json for automatic simulation discovery.",
    )
    parser.add_argument(
        "--cohorts",
        nargs="+",
        default=None,
        help="Optional cohort filter (e.g., PD_MALES or PD_MALES_PROXY).",
    )
    parser.add_argument(
        "--ora-databases",
        nargs="+",
        default=None,
        help=(
            "Override pathway databases for ORA (e.g., kegg reactome hallmark wikipathways "
            "biocarta pid go_bp progeny)."
        ),
    )
    parser.add_argument(
        "--attributed-pathways",
        nargs="*",
        default=None,
        help="Override explicit attributed pathway names (pass empty flag to clear explicit list).",
    )
    parser.add_argument(
        "--no-attributed-keywords",
        action="store_true",
        help="Disable automatic keyword-based pathway attribution.",
    )
    parser.add_argument(
        "--only-databases",
        action="store_true",
        help="Run ORA using only the selected pathway databases without any attributed pathway names or keywords.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(processed_dir / "step06_validation"),
        help="Output directory for Step 06 validation CSV tables.",
    )
    parser.add_argument("--reports-dir", default=str(reports_dir))
    parser.add_argument("--logging-level", default="INFO")
    return parser


def run_step06(args: argparse.Namespace) -> None:
    configure_logging(args.logging_level)
    config_path = Path(args.config)
    config = _resolve_step06_config(
        config_path=config_path,
        step04_config_path=Path(args.step04_config),
        cli_ora_databases=getattr(args, "ora_databases", None),
        cli_attributed_pathways=getattr(args, "attributed_pathways", None),
        cli_disable_keywords=bool(getattr(args, "no_attributed_keywords", False)),
        cli_only_databases=bool(getattr(args, "only_databases", False)),
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    step05_report_path = Path(args.step05_report)
    if not step05_report_path.exists():
        raise FileNotFoundError(
            f"Step 05 report not found at '{step05_report_path}'. Run Step 05 first."
        )

    with step05_report_path.open("r", encoding="utf-8") as handle:
        step05_report = json.load(handle)

    perturbation_node = str(
        step05_report.get("parameters", {}).get(
            "perturbation_node", config.get("kiss1r_gene", "KISS1R")
        )
    )

    adata = sc.read_h5ad(Path(args.dataset_path))
    pkn_universe, pkn_degrees = _reconstruct_omnipath_pkn_universe(step05_report)
    progeny_net, ora_pathway_net = _load_pathway_gene_sets(config)

    cohorts_in_step05: dict[str, Any] = step05_report.get("cohorts", {})
    if args.cohorts:
        allowed = {c.upper() for c in args.cohorts}
        cohorts_in_step05 = {
            k: v
            for k, v in cohorts_in_step05.items()
            if k.upper() in allowed or any(k.upper().startswith(a) for a in allowed)
        }

    report: dict[str, Any] = {
        "config_path": str(config_path),
        "step05_report_path": str(step05_report_path),
        "parameters": config,
        "pkn_universe_size": len(pkn_universe),
        "cohorts": {},
    }

    for step05_cohort_key, cohort_meta in cohorts_in_step05.items():
        adata_key, cohort_label, strat_mode = _parse_step05_cohort_key(
            step05_cohort_key, str(config["stratification_mode"])
        )
        LOGGER.info(
            "=== Step 06 Validation for %s (cohort=%s, stratification=%s) ===",
            step05_cohort_key,
            cohort_label,
            strat_mode,
        )

        _, sub_adata = _subset_adata_by_cohort(adata, adata_key)
        pdata_norm, strat_stats = make_pseudobulk_stratified(
            adata=sub_adata,
            cohort_label=cohort_label,
            config=config,
            min_cells=int(config["min_cells_pseudobulk"]),
            stratification_mode=strat_mode,
        )

        # Multi-granularity PROGENy validation on pseudobulk expression (High vs Low)
        progeny_df = run_progeny_footprint_validation(
            pdata_norm=pdata_norm,
            progeny_net=progeny_net,
            n_perm=int(config["progeny_permutations"]),
            seed=int(config["random_seed"]),
        )
        progeny_csv = out_dir / f"PROGENy_Footprints_{step05_cohort_key}.csv"
        progeny_df.to_csv(progeny_csv, index=False)

        simulations = cohort_meta.get("simulations", {})
        validated_states: dict[str, Any] = {}
        state_networks_for_jaccard: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}

        for state_key, sim_info in simulations.items():
            if sim_info.get("status") != "success":
                validated_states[state_key] = {
                    "status": sim_info.get("status", "skipped"),
                    "message": sim_info.get("message"),
                }
                continue

            target_state = int(sim_info.get("target_state", 1))
            edges_csv_path = Path(sim_info["edges_csv"])
            nodes_csv_path = (
                Path(sim_info["nodes_csv"]) if sim_info.get("nodes_csv") else None
            )

            if not edges_csv_path.exists():
                LOGGER.warning("Edges CSV missing at %s. Skipping state.", edges_csv_path)
                continue

            nodes_df, edges_df = _load_clean_carnival_tables(
                edges_csv_path,
                nodes_csv_path,
                pkn_universe,
                perturbation_node=perturbation_node,
            )
            state_networks_for_jaccard[target_state] = (nodes_df, edges_df)

            # 1. Two-Step Pathway Inference (ORA + Optional Stimulus-Attributed EA)
            ora_df, two_step_summary = run_two_step_pathway_inference(
                nodes_df=nodes_df,
                pkn_universe=pkn_universe,
                pathway_net=ora_pathway_net,
                attributed_pathways=config["attributed_pathways"],
                use_attributed_keywords=bool(config.get("use_attributed_keywords", True)),
                attributed_pathway_keywords=config.get("attributed_pathway_keywords"),
                target_state=target_state,
                fdr_thresh=float(config["fdr_threshold"]),
                n_perm_step2=int(config.get("step2_permutations", 10000)),
                seed=int(config["random_seed"]),
            )
            ora_csv = (
                out_dir
                / f"CARNIVAL_ORA_TwoStep_{step05_cohort_key}_state_{target_state}.csv"
            )
            ora_df.to_csv(ora_csv, index=False)

            # 2. Network Topology & 100 Degree-Preserving Randomizations
            topo_df, topo_summary = run_topological_and_randomization_analysis(
                nodes_df=nodes_df,
                edges_df=edges_df,
                pkn_degrees=pkn_degrees,
                n_random=int(config["n_random_networks"]),
                top_n=int(config["top_n_topology"]),
                seed=int(config["random_seed"]),
            )
            topo_csv = (
                out_dir
                / f"CARNIVAL_Topology_{step05_cohort_key}_state_{target_state}.csv"
            )
            topo_df.to_csv(topo_csv, index=False)

            # 3. Step 04 TF Stability Audit Integration
            tf_stab_summary = integrate_step04_tf_stability(
                nodes_df=nodes_df,
                step04_dir=Path(args.step04_dir),
                cohort_label=cohort_label,
                strat_mode=strat_mode,
            )

            validated_states[state_key] = {
                "status": "validated",
                "target_state": target_state,
                "source_edges_csv": str(edges_csv_path),
                "source_nodes_csv": str(nodes_csv_path) if nodes_csv_path else None,
                "ora_csv": str(ora_csv),
                "topology_csv": str(topo_csv),
                "two_step_pathway_inference": two_step_summary,
                "network_topology_and_randomization": topo_summary,
                "step04_tf_stability_in_network": tf_stab_summary,
            }

        # 4. Jaccard Similarity, Sign Inversion & Parsimony Comparison (state_1 vs state_-1)
        state_comparison = compare_opposite_target_states(
            state_networks_for_jaccard,
            perturbation_node=perturbation_node,
            canonical_effector=str(config.get("gnaq_gene", "GNAQ")),
        )

        report["cohorts"][step05_cohort_key] = {
            "cohort_label": cohort_label,
            "stratification_mode": strat_mode,
            "stratification": strat_stats,
            "progeny_pseudobulk_csv": str(progeny_csv),
            "progeny_pseudobulk_footprints": progeny_df.to_dict(orient="records"),
            "state_1_vs_state_minus1_comparison": state_comparison,
            "simulations_validation": validated_states,
        }

    report_path = reports_dir / "step06_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    LOGGER.info("Saved Step 06 validation report -> %s", report_path)