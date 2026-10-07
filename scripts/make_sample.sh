#!/usr/bin/env bash
# Regenerate the shipped sample dataset + reports deterministically.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-out/sample_tech}
python -m synthitd.cli generate --config configs/tech_small.yaml --out "$OUT"
python -m synthitd.cli benchmark --data "$OUT"
python -m synthitd.cli export-cert --data "$OUT" --out "$OUT/cert"
echo "Sample written to $OUT"
