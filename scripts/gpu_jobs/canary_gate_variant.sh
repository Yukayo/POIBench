#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
m="${1:?usage: GATE_PY=<python> bash scripts/gpu_jobs/canary_gate_variant.sh <method> <cfg>}"
cfg="${2:?usage: GATE_PY=<python> bash scripts/gpu_jobs/canary_gate_variant.sh <method> <cfg>}"
PY="${GATE_PY:?set GATE_PY to the method env python}"
case "$m" in
  getnext) var=GETNEXT_CFG ;;
  gnpr_sid) var=GNPR_SID_CFG ;;
  ds_llm4poi) var=DSLLM4POI_CFG ;;
  llm4poi) var=LLM4POI_CFG ;;
  gallm) var=GALLM_CFG ;;
  *) echo "canary_gate_variant: no configuration variable is mapped for method '$m'; refusing to run" >&2; exit 2 ;;
esac
for pol in H-100 H-10; do
  [ -f "exports/nyc/$pol/MANIFEST.json" ] || { echo "exports/nyc/$pol missing: run scripts/export_inputs.py" >&2; exit 4; }
  if [ "${GATE_SKIP_PREP:-0}" != "1" ]; then
    env "$var=$cfg" "$PY" -I "methods/$m/run.py" --ds nyc --policy "$pol" --seed 0 --canary-prep --cfg "$cfg"
  fi
  env "$var=$cfg" "$PY" -I -c "
import sys
sys.path.insert(0, '.')
from poibench.canary import run_canary
rep = run_canary('methods/$m/run.py:predict', 'nyc', '$pol', 'val', n=300, seed=0, tag='methods_${m}_run.py-${cfg}',
                 expect_config={'cfg': '$cfg'})
print('canary', '$m', '$cfg', '$pol', rep['result'], 'mismatches', rep.get('n_mismatch'), 'config', rep.get('adapter_config'),
      'config_check', rep.get('config_check'))
sys.exit(0 if rep['result'] == 'PASS' and rep.get('config_check') == 'PASS' else 1)
"
done
echo "canary gate $m cfg=$cfg: PASS (configuration used: $var=$cfg, verified in both runs)"
