#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
export PYTHONPATH=./src
exec ./facenet-venv/bin/python identify_speakers.py "$@"
