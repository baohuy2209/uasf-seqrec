"""Verify the published tables, or export them from the completed study. No models run."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "export"), nargs="?", default="verify")
    parser.add_argument("--destination", default="results/paper")
    parser.add_argument("--run-root", default="runs/memo0711_v3")
    parser.add_argument("--early-root", default="runs/checkpoint2_full_20261106")
    parser.add_argument("--metadata-root", default="artifacts/metadata")
    args = parser.parse_args()
    from taimma.paths import resolve_project_path
    from taimma.paper_release import export_bundle, verify_bundle

    destination = resolve_project_path(args.destination)
    if args.command == "verify":
        result = verify_bundle(destination)
    else:
        result = export_bundle(ROOT, resolve_project_path(args.run_root),
                               resolve_project_path(args.early_root),
                               resolve_project_path(args.metadata_root), destination)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
