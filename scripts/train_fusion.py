#!/usr/bin/env python3
"""Thin wrapper. The implementation lives in caddack.gnn.cli_train_fusion."""
from caddack.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["train-fusion", *__import__("sys").argv[1:]]))
