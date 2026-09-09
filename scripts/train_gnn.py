#!/usr/bin/env python3
"""Thin wrapper. The implementation lives in caddack.gnn.cli_train_gnn."""
from caddack.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["train-gnn", *__import__("sys").argv[1:]]))
