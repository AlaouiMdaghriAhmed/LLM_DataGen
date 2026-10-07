#!/usr/bin/env bash
# Regenerate the scaled reference dataset (finance, 1000x120) deterministically.
# The full event stream is large and gitignored; compact reports go under reference/.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-out/finance_reference}
python -m synthitd.cli generate --config configs/finance_r62_sparse.yaml --out "$OUT" --no-render
python -m synthitd.cli benchmark --data "$OUT"
echo "Reference dataset written to $OUT (benchmark.json, manifest.json)"
