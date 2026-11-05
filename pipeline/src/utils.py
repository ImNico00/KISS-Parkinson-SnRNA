from __future__ import annotations

import ast
import logging
import os
from pathlib import Path
from typing import Any

def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

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

def load_yaml_file(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for lineno, raw_line in enumerate(config_path.read_text(encoding="utf-8").splitlines(), start=1):
        # Pulisce eventuali caratteri NBSP (\u00a0) e spazi indesiderati
        raw_line = raw_line.replace("\u00a0", " ")
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if indent % 2 != 0:
            raise ValueError(f"Invalid indentation in YAML config at {config_path}:{lineno}")

        line = raw_line.strip()
        if ":" not in line:
            raise ValueError(f"Expected key/value mapping in YAML config at {config_path}:{lineno}")

        key, raw_value = line.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()

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
                    # Tenta di valutare liste inline (es. [0, 60, 70]) o altri letterali Python
                    try:
                        value = ast.literal_eval(raw_value)
                    except (ValueError, SyntaxError):
                        value = raw_value.strip("'\"")
        current[key] = value

    return root