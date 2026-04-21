#!/bin/bash
# Run classical baselines
cd "$(dirname "$0")"
source venv/bin/activate
echo "=== Running Classical Baselines ==="
python3 -m src.classical.classical_baselines
echo "=== Done ==="
