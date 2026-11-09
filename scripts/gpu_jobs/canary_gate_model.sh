#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
m="${1:?usage: GATE_PY=<python> bash scripts/gpu_jobs/canary_gate_model.sh <method> <key>}"
key="${2:?usage: GATE_PY=<python> bash scripts/gpu_jobs/canary_gate_model.sh <method> <key>}"
PY="${GATE_PY:?set GATE_PY to the method env python}"
case "$m" in
  ds_llm4poi) var=DSLLM4POI_MODEL ;;
  gnpr_sid) var=GNPR_SID_BACKBONE ;;
  *) echo "no backbone variable for $m" >&2; exit 2 ;;
esac
for pol in H-100 H-10; do
  [ -f "exports/nyc/$pol/MANIFEST.json" ] || { echo "exports/nyc/$pol missing: run scripts/export_inputs.py" >&2; exit 4; }
  env "$var=$key" "$PY" -I "methods/$m/run.py" --ds nyc --policy "$pol" --seed 0 --canary-prep
  env "$var=$key" "$PY" -I -c "
import sys
sys.path.insert(0, '.')
from poibench.canary import run_canary
rep = run_canary('methods/$m/run.py:predict', 'nyc', '$pol', 'val', n=300, seed=0, tag='methods_${m}_run.py-${key}')
print('canary', '$m', '$key', '$pol', rep['result'], 'mismatches', rep.get('n_mismatch'))
sys.exit(0 if rep['result'] == 'PASS' else 1)
"
done
echo "canary gate $m backbone=$key: PASS"
