#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p third_party
REPOS=(
  "GETNext https://github.com/songyangco/GETNext 6a79081aec6951ebec31bff6400d6ba0e40a613e"
  "Graph-Flashback https://github.com/kevin-xuan/Graph-Flashback 3489e9a52bf22c36efc54b1eb1e1dabc94d8a596"
  "Graph-Flashback-KGE https://github.com/kevin-xuan/KGE 9fd927301fe9f2d644716f8d2e89b1d4887e21aa"
  "DS-LLM4POI https://github.com/ryonsd/DS-LLM4POI f31ab6ffa3c76ae95d2a086b6eae38005a657a00"
  "LLM4POI https://github.com/neolifer/LLM4POI 83a44f4eece089828f58bae258b4692592eb3213"
  "GNPR-SID https://github.com/wds1996/GNPR-SID 641088f08b6f5750ae87bfcee850e152161121df"
  "GA-LLM https://github.com/liuzh368/GA-LLM 360f5c3e4d70b5191d45f8ef389a3fdb05bbcc9f"
  "MMPOI https://github.com/zzmylq/MMPOI 32f9c76e8768af3b268f0c95679bb7e0c739f766"
  "LLM-Augmented-POIRec https://github.com/wzehui/LLM-Augmented-POIRec c7dac9daf9dcde589f301b5b07148e4b20a2cd68"
)
ONLY="${1:-}"
for spec in "${REPOS[@]}"; do
  read -r name url commit <<< "$spec"
  if [ -n "$ONLY" ] && [ "$ONLY" != "$name" ]; then continue; fi
  dst="third_party/$name"
  if [ ! -d "$dst/.git" ]; then
    git clone --quiet "$url" "$dst"
  fi
  git -c safe.directory='*' -C "$dst" fetch --quiet origin "$commit" 2>/dev/null || true
  git -c safe.directory='*' -C "$dst" checkout --quiet "$commit"
  got="$(git -c safe.directory='*' -C "$dst" rev-parse HEAD)"
  if [ "$got" != "$commit" ]; then echo "$name: HEAD $got != pinned $commit" >&2; exit 1; fi
  echo "$name @ $commit: OK"
done
if [ -z "$ONLY" ] || [ "$ONLY" = "GETNext" ]; then
  bash methods/getnext/patches/fetch_upstream.sh
fi
