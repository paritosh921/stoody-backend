"""Cleanup expired private upload quarantine/rejected files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.upload_security.cleanup import cleanup_private_upload_storage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean expired private upload storage files.")
    parser.add_argument("--root", type=Path, default=None, help="Private upload root. Defaults to backend settings.")
    parser.add_argument("--rejected-retention-days", type=int, default=None)
    parser.add_argument("--quarantine-retention-hours", type=int, default=None)
    parser.add_argument("--execute", action="store_true", help="Delete files. Default is dry-run.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = cleanup_private_upload_storage(
        args.root,
        rejected_retention_days=args.rejected_retention_days,
        quarantine_retention_hours=args.quarantine_retention_hours,
        dry_run=not args.execute,
    )
    print(json.dumps(result.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
