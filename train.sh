#!/usr/bin/env bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m scae "$@"
