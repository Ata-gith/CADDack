#!/usr/bin/env python3
import argparse
import importlib
import pathlib
import sys


def main():
    parser = argparse.ArgumentParser(prog="caddack", description="CADDack CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Ensure repo root (containing 'scripts/') is importable
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    scripts_dir = repo_root / "scripts"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Load subcommands from scripts/*
    fs = importlib.import_module("scripts.fetch_structures")
    fs.add_cli(sub)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
