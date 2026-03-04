#!/usr/bin/env bash
# Run export_embeddings.py with facenet-venv and correct PYTHONPATH.
# Usage: ./run_export_embeddings.sh <model_dir> <data_dir> [options...]
# Example: ./run_export_embeddings.sh ~/models/facenet/20170216-091149 ~/datasets/lfw/mylfw

set -e
cd "$(dirname "$0")"
export PYTHONPATH=./src
exec ./facenet-venv/bin/python contributed/export_embeddings.py "$@"
