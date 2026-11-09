from __future__ import annotations
import os as _pb_os
def _pb_root():
    p = _pb_os.path.dirname(_pb_os.path.abspath(__file__))
    while not _pb_os.path.isfile(_pb_os.path.join(p, "poibench", "__init__.py")) and _pb_os.path.dirname(p) != p:
        p = _pb_os.path.dirname(p)
    return p
_PB_ROOT = _pb_os.environ.get("POIBENCH_ROOT") or _pb_root()
_PB_MODELS = _pb_os.environ.get("POIBENCH_MODEL_DIR") or _pb_os.path.join(_PB_ROOT, "models")

import glob
import heapq
import inspect
import json
import os
import time

import numpy as np

from methods.gnpr_sid.upstream import DEFAULT_BACKBONE, backbone

BACKBONES = {
    "llama3-8b": ("Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3-8B-Instruct",
                  "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"),
    "qwen2.5-7b": ("Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"),
    "qwen2.5-14b": ("Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-14B-Instruct", None),
    "qwen2.5-32b": ("Qwen2.5-32B-Instruct", "Qwen/Qwen2.5-32B-Instruct", None),
}
BACKBONE = backbone()
LLM_DIRNAME, LLM_REPO, LLM_REVISION = BACKBONES[BACKBONE]
SFT_CALL = dict(batch_size=8, num_train_epochs=5, learning_rate=2e-5, grad_accum=2, cutoff_len=2048)
_MB = int(os.environ.get("GNPR_SID_MICRO_BATCH", "8"))
if _MB == 0:
    raise RuntimeError("GNPR_SID_MICRO_BATCH=0: the memory probe found no micro-batch that fits (probe_mem.json)")
if _MB != 8:
    if 16 % _MB:
        raise ValueError(f"GNPR_SID_MICRO_BATCH={_MB} must divide 16")
    SFT_CALL = dict(SFT_CALL, batch_size=_MB, grad_accum=16 // _MB)
RESPONSE_HEADER = "### Response:\n"
NUM_BEAMS = 10
INFER_BATCH = int(os.environ.get("GNPR_SID_INFER_BATCH", "32"))


def base_model_path() -> str:
    p = os.path.join(os.environ.get("POIBENCH_MODEL_DIR", f"{_PB_MODELS}"), LLM_DIRNAME)
    if BACKBONE != DEFAULT_BACKBONE and not os.path.exists(os.path.join(p, ".poibench_complete")):
        x = os.environ.get("POIBENCH_MODEL_DIR_X")
        if x and os.path.exists(os.path.join(x, LLM_DIRNAME, ".poibench_complete")):
            return os.path.join(x, LLM_DIRNAME)
    return p


def base_model_revision() -> str | None:
    p = os.path.join(base_model_path(), ".poibench_complete")
    try:
        with open(p) as f:
            return json.load(f).get("sha")
    except Exception:
        return None


def load_tokenizer(path: str | None = None):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path or base_model_path(), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


class SIDTrie:

    def __init__(self, tok, pid2code: dict):
        head = tok(RESPONSE_HEADER, add_special_tokens=False)["input_ids"]
        self.children = [{}]
        self.leaf = [None]
        self.tokens = {}
        for pid in sorted(pid2code):
            full = tok(RESPONSE_HEADER + pid2code[pid], add_special_tokens=False)["input_ids"]
            if full[:len(head)] != head:
                raise RuntimeError(f"response header tokenizes differently before SID {pid2code[pid]}")
            seq = tuple(full[len(head):])
            self.tokens[int(pid)] = seq
            n = 0
            for t in seq:
                c = self.children[n].get(t)
                if c is None:
                    c = len(self.children)
                    self.children[n][t] = c
                    self.children.append({})
                    self.leaf.append(None)
                n = c
            if self.leaf[n] is not None:
                raise RuntimeError(f"two POIs share the SID token sequence {seq}")
            self.leaf[n] = int(pid)
        bad = [i for i, l in enumerate(self.leaf) if l is not None and self.children[i]]
        if bad:
            raise RuntimeError(f"SID token sequences are not prefix-free ({len(bad)} nodes)")
        self.max_depth = max(len(s) for s in self.tokens.values())
        self.allowed = [np.array(sorted(c), dtype=np.int64) for c in self.children]


def check_labels(trainer, tok, n_show: int = 1) -> dict:
    ds = trainer.train_dataset
    coll = trainer.data_collator
    n_empty, n_tok, shown = 0, [], []
    for s in range(0, len(ds), 64):
        feats = [{k: ds[i][k] for k in ("input_ids", "attention_mask")} for i in range(s, min(s + 64, len(ds)))]
        b = coll(feats)
        lab = b["labels"]
        m = lab != -100
        n_empty += int((m.sum(1) == 0).sum())
        n_tok += m.sum(1).tolist()
        if len(shown) < n_show:
            shown.append(tok.decode(lab[0][m[0]].tolist()))
    out = dict(n_samples=len(ds), n_all_masked=n_empty, label_tokens_mean=float(np.mean(n_tok)),
               label_tokens_min=int(np.min(n_tok)), example_label=shown,
               max_input_tokens=int(max(len(ds[i]["input_ids"]) for i in range(len(ds)))))
    if n_empty == len(ds):
        raise RuntimeError(f"every training sample has all labels masked: {out}")
    return out


def sft(samples: list, out_dir: str, max_steps: int | None = None, seed: int = 0, log=print) -> dict:
    done = os.path.join(out_dir, "SFT_DONE.json")
    if os.path.exists(done):
        with open(done) as f:
            log(f"SFT: reusing {out_dir}")
            return json.load(f)
    import torch
    from datasets import Dataset
    from transformers import set_seed
    from methods.gnpr_sid import upstream as U
    mod = U.sft_module()
    os.makedirs(out_dir, exist_ok=True)
    ckpts = sorted(glob.glob(os.path.join(out_dir, "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[1]))
    resume = ckpts[-1] if ckpts else None
    info = {}

    def before_train(trainer):
        info["labels"] = check_labels(trainer, trainer.tokenizer)
        log(f"SFT label check: {info['labels']}")

    if BACKBONE != DEFAULT_BACKBONE:
        mod._POIBENCH_EOS = load_tokenizer(base_model_path()).eos_token
        info["eos_suffix"] = mod._POIBENCH_EOS
        log(f"SFT backbone {BACKBONE}: format_fn suffix {mod._POIBENCH_EOS!r}")
    set_seed(seed)
    train_data = Dataset.from_list(samples)
    hooks = dict(before_train=before_train, resume_from_checkpoint=resume, save_total_limit=2)
    if max_steps:
        hooks["max_steps"] = int(max_steps)
    t0 = time.time()
    mod.train(base_model=base_model_path(), train_data=train_data, val_data=None, output_dir=out_dir,
              seed=seed, poibench_hooks=hooks, **SFT_CALL)
    wall = time.time() - t0
    hist = []
    st = sorted(glob.glob(os.path.join(out_dir, "checkpoint-*", "trainer_state.json")),
                key=lambda p: int(p.split("checkpoint-")[1].split(os.sep)[0]))
    if st:
        with open(st[-1]) as f:
            hist = json.load(f).get("log_history", [])
    sd = torch.load(os.path.join(out_dir, "final_sft", "adapter_model.bin"), map_location="cpu", weights_only=True)
    bnorm = float(sum(float(v.float().abs().sum()) for k, v in sd.items() if "lora_B" in k))
    n_b = sum(1 for k in sd if "lora_B" in k)
    meta = dict(n_samples=len(samples), sft_call=SFT_CALL, max_steps=max_steps, seed=seed,
                base_model=LLM_REPO, base_model_revision=base_model_revision(), resumed_from=resume,
                backbone=BACKBONE, eos_suffix=info.get("eos_suffix", "<|eot_id|>"),
                wall_s_this_attempt=wall, labels=info.get("labels"), lora_B_abs_sum=bnorm, n_lora_B=n_b,
                log_history=[h for h in hist if "loss" in h or "train_runtime" in h])
    if bnorm == 0.0:
        raise RuntimeError("LoRA B matrices are all zero after training: no gradient reached the adapters")
    with open(done + ".tmp", "w") as f:
        json.dump(meta, f, indent=1)
    os.replace(done + ".tmp", done)
    return meta


class Generator:

    def __init__(self, adapter_dir: str, pid2code: dict, device: str = "cuda"):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        self.torch = torch
        self.device = device
        self.tok = load_tokenizer(base_model_path())
        model = AutoModelForCausalLM.from_pretrained(base_model_path(), torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()
        self.model = model.to(device).eval()
        self.dtype = torch.bfloat16
        self.neg = torch.finfo(torch.bfloat16).min
        self.trie = SIDTrie(self.tok, pid2code)
        params = inspect.signature(self.model.forward).parameters
        self.keep_kw = "logits_to_keep" if "logits_to_keep" in params else (
            "num_logits_to_keep" if "num_logits_to_keep" in params else None)
        self.adapter_dir = adapter_dir

    def _view_cache(self, cache, b0: int, b1: int):
        from transformers import DynamicCache
        new = DynamicCache()
        new.key_cache = [k[b0:b1] for k in cache.key_cache]
        new.value_cache = [v[b0:b1] for v in cache.value_cache]
        new._seen_tokens = cache.get_seq_length()
        return new

    def _mask4d(self, allowed):
        torch = self.torch
        m = torch.zeros(allowed.shape, dtype=self.dtype, device=self.device)
        m.masked_fill_(~allowed, self.neg)
        return m[:, None]

    def _prefill(self, prompts):
        torch = self.torch
        enc = [self.tok(p, add_special_tokens=True)["input_ids"] for p in prompts]
        L = max(len(e) for e in enc)
        pad = self.tok.pad_token_id
        ids = torch.full((len(enc), L), pad, dtype=torch.long)
        mask = torch.zeros((len(enc), L), dtype=torch.long)
        for i, e in enumerate(enc):
            ids[i, L - len(e):] = torch.tensor(e)
            mask[i, L - len(e):] = 1
        ids, mask = ids.to(self.device), mask.to(self.device)
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        kw = dict(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True)
        if self.keep_kw:
            kw[self.keep_kw] = 1
        out = self.model(**kw)
        logp = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)
        return out.past_key_values, mask, logp, [len(e) for e in enc]

    def score_candidates(self, cache, mask, logp0, cand_lists, chunk: int = 4) -> list:
        torch = self.torch
        B, Lp = mask.shape
        out = []
        for c0 in range(0, B, chunk):
            c1 = min(B, c0 + chunk)
            per = []
            for b in range(c0, c1):
                seqs = [self.trie.tokens.get(int(v)) for v in cand_lists[b]]
                per.append(seqs)
            N = max(1, max(sum(len(s) - 1 for s in seqs if s) for seqs in per))
            nb = c1 - c0
            ids = np.full((nb, N), self.tok.pad_token_id, dtype=np.int64)
            depth = np.zeros((nb, N), dtype=np.int64)
            owner = np.full((nb, N), -1, dtype=np.int64)
            for r, seqs in enumerate(per):
                o = 0
                for ci, s in enumerate(seqs):
                    if not s or len(s) < 2:
                        continue
                    n = len(s) - 1
                    ids[r, o:o + n] = s[:-1]
                    depth[r, o:o + n] = np.arange(n)
                    owner[r, o:o + n] = ci
                    o += n
            same = (owner[:, :, None] == owner[:, None, :]) & (owner[:, :, None] >= 0)
            tri = np.tril(np.ones((N, N), dtype=bool))[None]
            new_allowed = (same & tri) | np.eye(N, dtype=bool)[None]
            allowed = torch.cat([mask[c0:c1, None, :].expand(nb, N, Lp).bool(),
                                 torch.from_numpy(new_allowed).to(self.device)], dim=2)
            base = mask[c0:c1].sum(1, keepdim=True)
            pos = base + torch.from_numpy(depth).to(self.device)
            sub = self._view_cache(cache, c0, c1)
            o = self.model(input_ids=torch.from_numpy(ids).to(self.device), attention_mask=self._mask4d(allowed),
                           position_ids=pos, past_key_values=sub, use_cache=True)
            logits = o.logits
            for r, seqs in enumerate(per):
                b = c0 + r
                res = np.full(len(seqs), -np.inf)
                lz = torch.logsumexp(logits[r].float(), dim=-1)
                o_ = 0
                firsts, idx_tok, idx_pos, spans, which = [], [], [], [], []
                for ci, s in enumerate(seqs):
                    if not s:
                        continue
                    n = len(s) - 1
                    firsts.append(s[0])
                    idx_pos += list(range(o_, o_ + n))
                    idx_tok += list(s[1:])
                    spans.append((len(idx_tok) - n, len(idx_tok)))
                    which.append(ci)
                    o_ += n
                if which:
                    tot = logp0[b, torch.tensor(firsts, device=self.device)].double()
                    if idx_tok:
                        p_ = torch.tensor(idx_pos, device=self.device)
                        t_ = torch.tensor(idx_tok, device=self.device)
                        lpt = (logits[r, p_, t_].float() - lz[p_]).double()
                        cs = torch.cat([torch.zeros(1, dtype=torch.float64, device=self.device), lpt.cumsum(0)])
                        a_ = torch.tensor([s_ for s_, _ in spans], device=self.device)
                        e_ = torch.tensor([e for _, e in spans], device=self.device)
                        tot = tot + (cs[e_] - cs[a_])
                    lens = torch.tensor([len(seqs[ci]) for ci in which], dtype=torch.float64, device=self.device)
                    res[which] = (tot / lens).cpu().numpy()
                out.append(res)
            del o, logits, sub
        return out

    def beam_search(self, cache, mask, logp0, K: int = NUM_BEAMS):
        torch = self.torch
        trie = self.trie
        B, Lp = mask.shape
        base_len = mask.sum(1)
        prompt_ok = mask.bool()
        finished = [[] for _ in range(B)]
        done = [False] * B
        beams = [[(0.0, 0)] for _ in range(B)]
        anc = None
        logp = logp0
        order = 0
        for step in range(trie.max_depth):
            r_idx, t_idx, meta = [], [], []
            kprev = len(beams[0])
            for b in range(B):
                if done[b]:
                    continue
                for s, (sc, node) in enumerate(beams[b]):
                    if node < 0:
                        continue
                    al = trie.allowed[node]
                    r_idx.append(np.full(len(al), b * kprev + s))
                    t_idx.append(al)
                    meta.append((b, s, sc, node, len(al)))
            if not meta:
                break
            r = torch.from_numpy(np.concatenate(r_idx)).to(self.device)
            t = torch.from_numpy(np.concatenate(t_idx)).to(self.device)
            vals = logp[r, t].double().cpu().numpy()
            off = 0
            per_q = {}
            for b, s, sc, node, n in meta:
                al = trie.allowed[node]
                ch = trie.children[node]
                lst = per_q.setdefault(b, [])
                for j in range(n):
                    tok = int(al[j])
                    lst.append((sc + float(vals[off + j]), s, tok, ch[tok]))
                off += n
            new_beams = [[(-np.inf, -1)] * K for _ in range(B)]
            parent = np.zeros((B, K), dtype=np.int64)
            next_tok = np.full((B, K), self.tok.pad_token_id, dtype=np.int64)
            cur_len = step + 1
            for b in range(B):
                if done[b] or b not in per_q:
                    done[b] = True
                    continue
                cands = sorted(per_q[b], key=lambda x: (-x[0], x[1], x[2]))[:2 * K]
                k = 0
                for rank, (sc, s, tok, child) in enumerate(cands):
                    if trie.leaf[child] is not None:
                        if rank >= K:
                            continue
                        norm = sc / cur_len
                        h = finished[b]
                        if len(h) < K:
                            heapq.heappush(h, (norm, -order, trie.leaf[child]))
                        elif norm > h[0][0]:
                            heapq.heapreplace(h, (norm, -order, trie.leaf[child]))
                        order += 1
                    elif k < K:
                        new_beams[b][k] = (sc, child)
                        parent[b, k] = s
                        next_tok[b, k] = tok
                        k += 1
                    if k >= K:
                        break
                live = [x for x in new_beams[b] if x[1] >= 0]
                if not live:
                    done[b] = True
                elif len(finished[b]) >= K and finished[b][0][0] >= max(x[0] for x in live) / cur_len:
                    done[b] = True
            if all(done):
                break
            par = torch.from_numpy(parent).to(self.device)
            if anc is None:
                anc_new = torch.zeros((B, K, 0), dtype=torch.bool, device=self.device)
            else:
                prev_cols = anc.shape[2]
                g = torch.gather(anc, 1, par[:, :, None].expand(B, K, prev_cols))
                own_prev = torch.zeros((B, K, prev_cols), dtype=torch.bool, device=self.device)
                own_prev.scatter_(2, (prev_cols - kprev + par)[:, :, None], True)
                anc_new = g | own_prev
            ncols = anc_new.shape[2]
            eye = torch.zeros((B, K, K), dtype=torch.bool, device=self.device)
            eye[:, torch.arange(K), torch.arange(K)] = True
            allowed_new = torch.cat([anc_new, eye], dim=2)
            dead = torch.from_numpy(np.array([[x[1] < 0 for x in nb] for nb in new_beams])).to(self.device)
            allowed_new = allowed_new & ~(dead[:, :, None] & ~torch.cat(
                [torch.zeros_like(anc_new), eye], dim=2))
            allowed = torch.cat([prompt_ok[:, None, :].expand(B, K, Lp), allowed_new], dim=2)
            pos = (base_len + step)[:, None].expand(B, K)
            ids = torch.from_numpy(next_tok).to(self.device)
            o = self.model(input_ids=ids, attention_mask=self._mask4d(allowed), position_ids=pos,
                           past_key_values=cache, use_cache=True)
            cache = o.past_key_values
            logp = torch.log_softmax(o.logits.float(), dim=-1).reshape(B * K, -1)
            anc = torch.cat([anc_new, torch.zeros((B, K, K), dtype=torch.bool, device=self.device)], dim=2)
            beams = new_beams
        res = []
        for b in range(B):
            h = sorted(finished[b], key=lambda x: (-x[0], -x[1]))
            res.append([(int(v), float(s)) for s, _, v in h])
        return res

    def _batch(self, prompts: list, cand_lists: list, log=None) -> list:
        torch = self.torch
        oom = False
        try:
            cache, mask, logp0, _ = self._prefill(prompts)
            cs = self.score_candidates(cache, mask, logp0, cand_lists)
            beams = self.beam_search(cache, mask, logp0)
            res = list(zip(beams, cs))
            del cache
            return res
        except torch.cuda.OutOfMemoryError:
            if len(prompts) == 1:
                raise
            oom = True
        cache = mask = logp0 = cs = beams = None
        torch.cuda.empty_cache()
        h = len(prompts) // 2
        self.oom_splits = getattr(self, "oom_splits", 0) + 1
        if log:
            log(f"CUDA OOM in a batch of {len(prompts)} queries: split into {h} + {len(prompts) - h} "
                f"(oom_splits={self.oom_splits})")
        return self._batch(prompts[:h], cand_lists[:h], log) + self._batch(prompts[h:], cand_lists[h:], log)

    def run(self, prompts: list, cand_lists: list, log=None) -> list:
        torch = self.torch
        lens = [len(self.tok(p, add_special_tokens=True)["input_ids"]) for p in prompts]
        order = sorted(range(len(prompts)), key=lambda i: (lens[i], i))
        out = [None] * len(prompts)
        t0 = time.time()
        with torch.inference_mode():
            for bi in range(0, len(order), INFER_BATCH):
                ids = order[bi:bi + INFER_BATCH]
                for j, r in enumerate(self._batch([prompts[i] for i in ids], [cand_lists[i] for i in ids], log)):
                    out[ids[j]] = r
                if log and (bi // INFER_BATCH) % 20 == 0:
                    log(f"infer {bi + len(ids)}/{len(prompts)} queries, {time.time() - t0:.0f} s")
        return out
