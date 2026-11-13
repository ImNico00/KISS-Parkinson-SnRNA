from __future__ import annotations

import ast
import logging
import os
from pathlib import Path
from typing import Any
import h5py
import numpy as np
import scipy.sparse as sp
import anndata as ad
import pandas as pd
import re

def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

def _normalize_sample_id_index(raw_index: pd.Index) -> pd.Index:
    """
    Normalizes raw H5AD sample identifiers to match SAMPLE.csv ASAP_sample_id / sample_id:
      - Fixes typo 'ASAP_PMBDS_' -> 'ASAP_PMDBS_' (case-insensitive)
      - Strips replicate suffixes after sample number (e.g. '_s001_Rep1', '_s003_rep1', '_s001_1')
      - Strips trailing '_Rep1' / '_rep1' on non-ASAP sample IDs
    """
    return (
        raw_index.astype(str)
        .str.strip()
        .str.replace(r"(?i)^ASAP_PMBDS_", "ASAP_PMDBS_", regex=True)
        .str.replace(r"(?i)(_s\d+)_(?:rep\d+|\d+)$", r"\1", regex=True)
        .str.replace(r"(?i)_(?:rep\d+)$", "", regex=True)
    )

def match_regions(series: pd.Series, targets: list[str]) -> pd.Series:
    values = series.fillna("").astype(str).str.strip().str.casefold()
    mask = pd.Series(False, index=series.index)

    for target in targets:
        candidates = [
            target.split(" (")[0],
            *re.findall(r"\(([^,]+),", target),
            *re.findall(r"UBERON:\d+", target),
        ]

        for candidate in candidates:
            candidate = candidate.strip().casefold()
            if candidate:
                mask |= values.str.contains(re.escape(candidate), na=False)

    return mask

def _sanitize_obs_for_h5ad(obs: pd.DataFrame) -> pd.DataFrame:
    """
    Converts mixed object-dtype columns in adata.obs into clean strings or numeric types
    so h5py/anndata never fails with TypeError during write_h5ad or write_elem.
    """
    clean_obs = obs.copy()
    clean_obs.index = clean_obs.index.astype(str)

    for col in clean_obs.columns:
        series = clean_obs[col]
        if isinstance(series.dtype, pd.CategoricalDtype):
            clean_obs[col] = series.cat.remove_unused_categories()
        elif series.dtype == object or pd.api.types.is_string_dtype(series.dtype):
            # Try numeric conversion only if all non-null values are numeric
            non_null = series.dropna()
            if not non_null.empty:
                numeric_try = pd.to_numeric(non_null, errors="coerce")
                if numeric_try.notna().all():
                    clean_obs[col] = pd.to_numeric(series, errors="coerce").astype(float)
                    continue
            clean_obs[col] = series.where(series.notna(), "").astype(str)

    return clean_obs

def _overwrite_obs_in_h5ad(h5ad_path: Path, obs_df: pd.DataFrame) -> None:
    """Updates only the /obs table inside an existing .h5ad file without rewriting .X."""
    from anndata.io import write_elem

    with h5py.File(h5ad_path, "a") as f:
        write_elem(f, "obs", obs_df)

def _first_non_null(series: pd.Series) -> Any:
    """Returns the first non-null entry in a Series, or np.nan if all are null."""
    valid = series.dropna()
    return valid.iloc[0] if not valid.empty else np.nan

def ensure_parent_dir(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target

def load_dotenv_file(path: str | Path) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value

def _extract_1d_expr(adata: ad.AnnData, gene: str) -> np.ndarray:
    """Extracts a flattened 1D numpy array of expression values for a single gene."""
    x = adata[:, gene].X
    if sp.issparse(x):
        return x.toarray().ravel()
    return np.asarray(x).ravel()

def _ensure_list(val: Any) -> list[Any]:
    """Converts stringified lists from config into standard Python lists."""
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val.strip())
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
        except (ValueError, SyntaxError):
            return [x.strip() for x in val.split(",") if x.strip()]
    return list(val)

def build_tagged_path(
    base_path: str | Path,
    *tags: object,
    default_dir: str | Path | None = None,
    sep: str = "_",
    default_ext: str = "",
) -> Path:
    """
    Inserisce uno o più tag prima dell'estensione di qualsiasi file o percorso.

    Esempi:
      build_tagged_path("qc.png", "MALES", default_dir="/plots")
        -> Path("/plots/qc_MALES.png")
      build_tagged_path("custom/volcano.pdf", "FEMALES", "Astro")
        -> Path("custom/volcano_FEMALES_Astro.pdf")
      build_tagged_path("edgeR.csv", "ALL", "PDvsControl", "Dopaminergic")
        -> Path("edgeR_ALL_PDvsControl_Dopaminergic.csv")
    """
    path = Path(base_path)

    # Separa il nome dal punto dell'estensione (es. "file.png" -> stem="file", ext=".png")
    stem = path.stem
    ext = path.suffix or (
        f".{default_ext.lstrip('.')}" if default_ext else ""
    )

    # Unisce tutti i tag validi passati alla funzione
    clean_tags = [str(t).strip(sep) for t in tags if t is not None and str(t).strip(sep)]
    tag_suffix = f"{sep}{sep.join(clean_tags)}" if clean_tags else ""

    new_filename = f"{stem}{tag_suffix}{ext}"

    # Se l'utente ha passato solo il nome del file senza cartelle, usa default_dir (se fornita)
    if default_dir is not None and path.parent == Path("."):
        return Path(default_dir) / new_filename

    return path.with_name(new_filename)

def _strip_yaml_comment(line: str) -> str:
    """Removes ‘#’ comments at the end of a line or across an entire line, while preserving ‘#’ characters inside quotation marks."""
    in_single = False
    in_double = False
    escape = False
    for idx, ch in enumerate(line):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return line[:idx].rstrip()
    return line.rstrip()

def load_yaml_file(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    lines = config_path.read_text(encoding="utf-8").splitlines()
    idx = 0
    total_lines = len(lines)

    while idx < total_lines:
        lineno = idx + 1
        raw_line = lines[idx].replace("\u00a0", " ")
        idx += 1

        # Removes inline comments and skips empty lines or lines containing only comments
        clean_line = _strip_yaml_comment(raw_line)
        if not clean_line.strip():
            continue

        indent = len(clean_line) - len(clean_line.lstrip(" "))
        if indent % 2 != 0:
            raise ValueError(f"Invalid indentation in YAML config at {config_path}:{lineno}")

        line = clean_line.strip()
        if ":" not in line:
            raise ValueError(f"Expected key/value mapping in YAML config at {config_path}:{lineno}")

        key, raw_value = line.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()

        # Supports lists [...] spanning multiple lines (including internal comments)
        if raw_value.startswith("[") and raw_value.count("[") > raw_value.count("]"):
            parts = [raw_value]
            balance = raw_value.count("[") - raw_value.count("]")
            while idx < total_lines and balance > 0:
                sub_raw = lines[idx].replace("\u00a0", " ")
                idx += 1
                sub_clean = _strip_yaml_comment(sub_raw).strip()
                if not sub_clean:
                    continue
                balance += sub_clean.count("[") - sub_clean.count("]")
                parts.append(sub_clean)
            raw_value = " ".join(parts)

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]

        if not raw_value:
            child: dict[str, Any] = {}
            current[key] = child
            stack.append((indent, child))
            continue

        value: Any
        lowered = raw_value.lower()
        if lowered in {"true", "false"}:
            value = lowered == "true"
        else:
            try:
                value = int(raw_value)
            except ValueError:
                try:
                    value = float(raw_value)
                except ValueError:
                    # Attempts to evaluate inline/multiline lists or other Python literals
                    try:
                        value = ast.literal_eval(raw_value)
                    except (ValueError, SyntaxError):
                        value = raw_value.strip("'\"")
        current[key] = value

    return root