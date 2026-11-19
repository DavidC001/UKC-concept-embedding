#!/bin/bash
KGHOME=$(pwd)
export PYTHONPATH="$KGHOME:${PYTHONPATH:-}"
export LOG_DIR="$(cd "$KGHOME/.." && pwd)/data/KGEmodels/standalone"
# KGE datasets live in <repo>/data/graph (built by data_build/build_ukc.py; benchmarks in data/graph/benchmarks)
export DATA_PATH="$(cd "$KGHOME/.." && pwd)/data/graph"
# source hyp_kg_env/bin/activate
