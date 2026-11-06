from pathlib import Path
import sys

from pipeline.src.carnival import build_step05_parser, run_step05

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

def main() -> None:
    parser = build_step05_parser(_REPO_ROOT)
    args = parser.parse_args()
    run_step05(args)

if __name__ == "__main__":
    main()