#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
m="${1:?usage: GATE_PY=<python> bash scripts/gpu_jobs/canary_gate.sh <method>}"
PY="${GATE_PY:?set GATE_PY to the method env python}"
for pol in H-100 H-10; do
  [ -f "exports/nyc/$pol/MANIFEST.json" ] || { echo "exports/nyc/$pol missing: run scripts/export_inputs.py" >&2; exit 4; }
  "$PY" -I "methods/$m/run.py" --ds nyc --policy "$pol" --seed 0 --canary-prep
  "$PY" -I scripts/canary.py --entry "methods/$m/run.py:predict" --ds nyc --policy "$pol" --split val --n 300
done
echo "canary gate $m: PASS"
