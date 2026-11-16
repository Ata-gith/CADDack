#!/usr/bin/env python3
"""Thin wrapper. The implementation lives in caddack.docking.cli_dock."""
import sys

from caddack.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["dock", *sys.argv[1:]]))
