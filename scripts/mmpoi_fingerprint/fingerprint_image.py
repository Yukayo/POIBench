import io
import json
import os
import sys
import tarfile
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.join(ROOT, "work", "mmpoi", "fingerprint")
OUT = os.path.join(HERE, "fp_imgs")
MP = os.path.join(HERE, "hf/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/"
                  "e8c3b32edf5434bc2275fc9bab85f82640a19130")
PROBE = ["49bbd6c0f964a520f4531fe3", "4a43c0aef964a520c6a61fe3", "4ab966c3f964a5203c7f20e3",
         "4ab5320cf964a5202b7320e3", "4c0ab56f7e3fc9288c1df482", "3fd66200f964a52000e71ee3"]


def load_emb(name):
    z = zipfile.ZipFile(f"{ROOT}/data_raw/mmpoi/processed_dl/NYC_processed.zip")
    d = {}
    for ln in z.read(f"NYC/{name}").split(b"\n"):
        if ln.strip():
            d.update(json.loads(ln))
    return d


def extract():
    os.makedirs(OUT, exist_ok=True)
    want = set(PROBE)
    zp = f"{ROOT}/data_raw/mmpoi/archives/NYC_WWW2024.zip"
    n = 0
    with zipfile.ZipFile(zp) as z, z.open("NYC/image.tgz") as tz, tarfile.open(fileobj=tz, mode="r|gz") as tf:
        for m in tf:
            if not m.isreg():
                continue
            base = os.path.basename(m.name)
            parts = base[:-4].split("_")
            if len(parts) == 4 and parts[2] in want:
                with open(os.path.join(OUT, base), "wb") as f:
                    f.write(tf.extractfile(m).read())
                n += 1
    print("extracted", n)


def mpnet():
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MP)
    mod = AutoModel.from_pretrained(MP).eval()

    def emb(texts):
        b = tok(texts, padding=True, truncation=True, max_length=384, return_tensors="pt")
        with torch.no_grad():
            h = mod(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        v = (h * m).sum(1) / m.sum(1)
        return torch.nn.functional.normalize(v, dim=-1).numpy()
    return emb


def captioner(spec):
    import torch
    name, *opts = spec.split("|")
    gk = {}
    prompt = None
    for o in opts:
        k, v = o.split("=", 1)
        if k == "prompt":
            prompt = v
        else:
            gk[k] = int(v)
    from PIL import Image
    from transformers import AutoProcessor
    if "blip2" in name.lower():
        from transformers import Blip2ForConditionalGeneration, Blip2Processor
        proc = Blip2Processor.from_pretrained(name)
        DT = getattr(torch, os.environ.get("FP_DTYPE", "bfloat16")); model = Blip2ForConditionalGeneration.from_pretrained(name, dtype=DT).eval()

        def cap(paths):
            ims = [Image.open(p).convert("RGB") for p in paths]
            b = proc(images=ims, return_tensors="pt")
            b["pixel_values"] = b["pixel_values"].to(DT)
            with torch.no_grad():
                out = model.generate(**b, **gk)
            return [s.strip() for s in proc.batch_decode(out, skip_special_tokens=True)]
        return cap
    if "blip" in name.lower() or "git" in name.lower():
        from transformers import AutoModelForImageTextToText as M
        try:
            from transformers import BlipForConditionalGeneration
            if "blip-image" in name:
                M = BlipForConditionalGeneration
        except ImportError:
            pass
        proc = AutoProcessor.from_pretrained(name)
        model = M.from_pretrained(name).eval()

        def cap(paths):
            ims = [Image.open(p).convert("RGB") for p in paths]
            if prompt:
                b = proc(images=ims, text=[prompt] * len(ims), return_tensors="pt")
            else:
                b = proc(images=ims, return_tensors="pt")
            with torch.no_grad():
                out = model.generate(**b, **gk)
            return [s.strip() for s in proc.batch_decode(out, skip_special_tokens=True)]
        return cap
    from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel
    proc = AutoImageProcessor.from_pretrained(name)
    tok = AutoTokenizer.from_pretrained(name)
    model = VisionEncoderDecoderModel.from_pretrained(name).eval()

    def cap(paths):
        ims = [Image.open(p).convert("RGB") for p in paths]
        px = proc(images=ims, return_tensors="pt").pixel_values
        with torch.no_grad():
            out = model.generate(px, **gk)
        return [s.strip() for s in tok.batch_decode(out, skip_special_tokens=True)]
    return cap


def run(models):
    tgt = load_emb("modal_image_embedding.json")
    emb = mpnet()
    files = sorted(os.listdir(OUT))
    for spec in models:
        cap = captioner(spec)
        for v in PROBE:
            fs = sorted([f for f in files if f.split("_")[2] == v], key=lambda f: int(f[:-4].split("_")[3]))
            caps = cap([os.path.join(OUT, f) for f in fs])
            E = emb(caps)
            t = np.asarray(tgt[v], dtype=float)
            mv = E.mean(0)
            joined = emb([" ".join(caps)])[0]
            print(json.dumps({"model": spec, "venue": v, "n": len(caps),
                              "cos_mean": round(float(mv @ t / np.linalg.norm(mv)), 4),
                              "cos_joined": round(float(joined @ t), 4),
                              "cos_single_max": round(float((E @ t).max()), 4),
                              "cos_first": round(float(E[0] @ t), 4), "caps": caps[:4]}), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", os.path.join(HERE, "hf"))
    if sys.argv[1] == "extract":
        extract()
    else:
        run(sys.argv[2:])
