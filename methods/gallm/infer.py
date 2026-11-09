import argparse
import hashlib
import json
import os
import sys
import time


def load_model(a):
    import math
    import torch
    import transformers
    from transformers import BitsAndBytesConfig
    from peft import PeftModel
    sys.path.insert(0, a.src)
    os.chdir(a.src)
    import test_new_fusion_gps as T
    from llama_attn_replace_sft import replace_llama_attn
    from geo_model.fusion_geo_encoder import FusionGeoEncoder4096 as FusionGeoEncoder

    device = "cuda:0"
    torch.cuda.set_device(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(a.base_model, model_max_length=a.seq_len,
                                                           padding_side="right", use_fast=True)
    replace_llama_attn(inference=True)
    config = transformers.AutoConfig.from_pretrained(a.base_model)
    context_size = a.context_size if a.context_size > 0 else a.seq_len
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    model = transformers.AutoModelForCausalLM.from_pretrained(
        a.base_model, config=config, torch_dtype=torch.bfloat16, device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, llm_int8_threshold=6.0,
                                               llm_int8_has_fp16_weight=False, bnb_4bit_compute_dtype=torch.bfloat16,
                                               bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4"))
    gps_mapping = T.load_gps_mapping(a.gps_mapping)
    gps_token_start_id = max(tokenizer.get_vocab().values()) + 1
    gps_id_list = list(gps_mapping.keys())
    gps_tokens = [f"<GPS {i}>" for i in range(len(gps_mapping))]
    tokenizer.add_special_tokens({"additional_special_tokens": gps_tokens})
    model.resize_token_embeddings(len(tokenizer))
    enc = FusionGeoEncoder(gps_embed_dim=128, quadkey_length=25, n=6, vocab_size=4096, fourier_output_dim=128,
                           final_dim=4096)
    enc.load_state_dict(torch.load(a.encoder, map_location=device))
    enc.to(device)
    enc.eval()
    T.apply_fusion_embedding_hook(model, gps_mapping, gps_id_list, gps_token_start_id, enc)
    tp = os.path.join(a.stage2, "trainable_params.bin")
    matched = None
    if os.path.isfile(tp):
        sd = torch.load(tp, map_location=model.device)
        r = model.load_state_dict(sd, strict=False)
        matched = len(sd) - len(r.unexpected_keys)
        print(f"trainable_params.bin: {len(sd)} tensors, {matched} matched the base model (upstream strict=False)",
              flush=True)
    model = PeftModel.from_pretrained(model, a.stage2, device_map="auto", torch_dtype=torch.float16)
    T.smart_tokenizer_and_embedding_resize({"pad_token": T.DEFAULT_PAD_TOKEN, "eos_token": T.DEFAULT_EOS_TOKEN,
                                            "bos_token": T.DEFAULT_BOS_TOKEN, "unk_token": T.DEFAULT_UNK_TOKEN},
                                           tokenizer, model)
    model.eval()
    gen = transformers.GenerationConfig(max_new_tokens=a.max_new_tokens, do_sample=True, temperature=0.6, top_k=40,
                                        top_p=0.1, eos_token_id=tokenizer.eos_token_id,
                                        pad_token_id=tokenizer.eos_token_id, num_return_sequences=1)
    return model, tokenizer, gen, matched, device


def generate_one(model, tokenizer, gen, device, prompt, qid, seed, seq_len=32768):
    import torch
    t1 = time.perf_counter()
    ntok = len(tokenizer.tokenize(prompt))
    if ntok >= seq_len:
        return dict(query_id=int(qid), text="", n_prompt_tokens=ntok, skipped=True, gen_s=0.0)
    enc_in = tokenizer(prompt, return_tensors="pt").to(device)
    torch.manual_seed(seed * 1_000_003 + int(qid))
    with torch.no_grad():
        outputs = model.generate(**enc_in, generation_config=gen)
    torch.cuda.empty_cache()
    text = tokenizer.decode(outputs[0, enc_in.input_ids.shape[1]:], skip_special_tokens=True)
    return dict(query_id=int(qid), text=text, n_prompt_tokens=int(enc_in.input_ids.shape[1]), skipped=False,
                gen_s=time.perf_counter() - t1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=4)
    ap.add_argument("--backend", default="hf", choices=["hf", "mock"])
    ap.add_argument("--base_model")
    ap.add_argument("--stage2")
    ap.add_argument("--encoder")
    ap.add_argument("--gps_mapping")
    ap.add_argument("--seq_len", type=int, default=32768)
    ap.add_argument("--context_size", type=int, default=32768)
    a = ap.parse_args()
    import pandas as pd
    df = pd.read_parquet(a.inp)
    n = len(df)
    lo, hi = a.shard * n // a.nshards, (a.shard + 1) * n // a.nshards
    df = df.iloc[lo:hi]
    keyed = "key" in df.columns
    want = dict(zip(df["query_id"].astype(int), df["key"])) if keyed else {}
    done = set()
    if os.path.exists(a.out):
        keep = []
        for line in open(a.out):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("kind") != "meta":
                q = int(r["query_id"])
                if keyed and q in want and r.get("key") != want[q]:
                    continue
                if not keyed or q in want:
                    done.add(q)
                keep.append(line if line.endswith("\n") else line + "\n")
        with open(a.out + ".tmp", "w") as f:
            f.writelines(keep)
        os.replace(a.out + ".tmp", a.out)

    def rec_of(qid, d):
        return dict(d, key=want[qid]) if keyed else d

    fo = open(a.out, "a")
    if a.backend == "mock":
        fo.write(json.dumps(dict(kind="meta", load_s=0.0, trainable_keys_matched=None, peak_gpu_mem_gb=None)) + "\n")
        for qid, p in zip(df["query_id"], df["prompt"]):
            if int(qid) in done:
                continue
            h = int(hashlib.sha256(p.encode()).hexdigest()[:8], 16)
            fo.write(json.dumps(rec_of(int(qid), dict(query_id=int(qid), text=str(h % 997), n_prompt_tokens=len(p) // 4,
                                                      skipped=False, gen_s=0.0))) + "\n")
        fo.close()
        return 0

    t0 = time.perf_counter()
    model, tokenizer, gen, matched, device = load_model(a)
    import torch
    load_s = time.perf_counter() - t0
    fo.write(json.dumps(dict(kind="meta", load_s=load_s, trainable_keys_matched=matched, peak_gpu_mem_gb=None)) + "\n")
    fo.flush()
    k = 0
    for qid, p in zip(df["query_id"], df["prompt"]):
        qid = int(qid)
        if qid in done:
            continue
        rec = generate_one(model, tokenizer, gen, device, p, qid, a.seed, a.seq_len)
        fo.write(json.dumps(rec_of(qid, rec)) + "\n")
        k += 1
        if k % 200 == 0:
            fo.flush()
            print(f"[infer shard {a.shard}] {k} done, {time.perf_counter() - t0:.0f} s", flush=True)
    peak = torch.cuda.max_memory_allocated() / 1e9
    fo.write(json.dumps(dict(kind="meta", load_s=None, trainable_keys_matched=matched, peak_gpu_mem_gb=peak)) + "\n")
    fo.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
