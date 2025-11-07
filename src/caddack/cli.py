#!/usr/bin/env python3
import argparse
import importlib
import pathlib
import sys


def main():
    parser = argparse.ArgumentParser(prog="caddack", description="CADDack CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # make repo root (containing 'scripts/') importable
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # subcommands
    import scripts.fetch_structures as fs
    fs.add_cli(sub)

    import scripts.run_qsar as qs
    qs.add_cli(sub)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
