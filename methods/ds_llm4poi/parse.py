from __future__ import annotations

import json
import re

FENCE_JSON = re.compile(r'```json\s*([\s\S]*?)```')
FENCE_ANY = re.compile(r'```[a-zA-Z]*\s*([\s\S]*?)```')
PLACE_ID_RE = re.compile(r'["\']?place_id["\']?\s*[:=]\s*["\']?\s*(\d+)')


def _int_id(v, allow_str: bool):
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if allow_str and isinstance(v, str) and v.strip().isdigit():
        return int(v.strip())
    return None


def parse_strict(text: str):
    m = FENCE_JSON.search(text or "")
    if not m:
        return None
    try:
        res = json.loads(m.group(1).strip())
        pid = res["place_id"]
        res["place_category"]
    except Exception:
        return None
    return _int_id(pid, allow_str=False)


def _first_json_object(text: str):
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text, m.start())
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def parse_lenient(text: str):
    text = text or ""
    for m in FENCE_ANY.finditer(text):
        try:
            obj = json.loads(m.group(1).strip())
        except ValueError:
            obj = _first_json_object(m.group(1))
        if isinstance(obj, dict) and "place_id" in obj:
            v = _int_id(obj["place_id"], allow_str=True)
            if v is not None:
                return v
    obj = _first_json_object(text)
    if isinstance(obj, dict) and "place_id" in obj:
        v = _int_id(obj["place_id"], allow_str=True)
        if v is not None:
            return v
    m = PLACE_ID_RE.search(text)
    return int(m.group(1)) if m else None


def ranked_list(answer, candidates, n_venues: int, k: int = 50) -> tuple[list, bool]:
    valid = answer is not None and 0 <= int(answer) < n_venues
    out, seen = [], set()
    for v in ([int(answer)] if valid else []) + [int(x) for x in candidates]:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out[:k], valid
