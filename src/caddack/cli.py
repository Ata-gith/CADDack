#!/usr/bin/env python3
"""CADDack command-line entry point.

Every subcommand lives inside the installed package, so the CLI works from a
wheel. Nothing here reaches outside the distribution -- an earlier version put
``scripts/`` on ``sys.path`` and imported from it, which made all subcommands
unusable once installed, because ``scripts/`` ships in neither the wheel nor the
sdist.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Construct the full CLI. Kept separate so tests can inspect it."""
    parser = argparse.ArgumentParser(prog="caddack", description="CADDack CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Imported here rather than at module scope so that `caddack --help` does not
    # pay for pandas/torch, and so a broken optional dependency cannot take down
    # unrelated subcommands.
    from caddack.fetch import fetch_structures
    from caddack.gnn import cli_train_fusion, cli_train_gnn
    from caddack.qsar import run_qsar_descriptors, train_qsar

    for module in (
        fetch_structures,
        run_qsar_descriptors,
        train_qsar,
        cli_train_gnn,
        cli_train_fusion,
    ):
        module.add_cli(sub)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
