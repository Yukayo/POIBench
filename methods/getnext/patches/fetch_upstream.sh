#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
COMMIT=6a79081aec6951ebec31bff6400d6ba0e40a613e
DST=third_party/GETNext
if [ ! -d "$DST/.git" ]; then
  git clone https://github.com/songyangco/GETNext.git "$DST"
fi
git -c safe.directory='*' -C "$DST" fetch --depth 1 origin "$COMMIT" 2>/dev/null || true
git -c safe.directory='*' -C "$DST" checkout -q "$COMMIT"
sha256sum -c - <<SUMS
3e439604bdaa2b2e770ae67aeace089264d22b4226397b39b68c0ada8eb61976  $DST/model.py
04369655005b3242d1064d5f27bbf1935b016295029b2088b406a33138b4151b  $DST/utils.py
97ed58a1e987d8e1f60a7dbcfc145bcb938f73e8101854cdb6d018803d5c6c83  $DST/build_graph.py
eec2bfb0b8edac4bcce643f883f3fb6905aec88c750681ff9c24c5e31d92aaa5  $DST/train.py
79f9d6c29c9f4c8cbe9d616805b5c466dd2b0766eede034a73c712cce1b92caf  $DST/param_parser.py
SUMS
echo "GETNext at $COMMIT: OK"
