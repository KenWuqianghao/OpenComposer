#!/bin/bash
set -euo pipefail

echo "=== Building OpenComposer sandbox Docker image ==="
cd "$(dirname "$0")"
docker build -t opencomposer/sandbox:latest -f Dockerfile.sandbox .
echo "=== Sandbox image built successfully ==="

echo ""
echo "For SWE-bench environments, you can optionally pull pre-built images:"
echo "  pip install swebench"
echo "  python -m swebench.harness.prepare_images --dataset_name SWE-bench/SWE-bench_Lite --split test"
echo ""
echo "This step is optional -- builtin tasks work without SWE-bench Docker images."
