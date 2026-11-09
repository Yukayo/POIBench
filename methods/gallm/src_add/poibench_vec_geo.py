import torch
from nltk import ngrams
from geo_model.fusion_geo_encoder import latlng2quadkey

_FEAT = {}


def _base(enc):
    return enc.get_base_model() if hasattr(enc, "get_base_model") else enc


def vec_fusion(enc, coords):
    m = _base(enc)
    p = next(m.parameters())
    idx, dig = [], []
    for lat, lng in coords:
        k = (m.quadkey_length, m.n, m.fourier_encoder.M, float(lat), float(lng))
        f = _FEAT.get(k)
        if f is None:
            qk = latlng2quadkey(lat, lng, m.quadkey_length)
            ng = [m.permutations_dict.get("".join(x), 0) for x in ngrams(qk, m.n)]
            qf = latlng2quadkey(lat, lng, level=m.fourier_encoder.M)
            f = (ng, [int(c) for c in qf])
            _FEAT[k] = f
        idx.append(f[0])
        dig.append(f[1])
    idx_t = torch.tensor(idx, dtype=torch.long).to(p.device)
    emb_geo = m.gps_encoder(m.gps_embed_model(idx_t))
    x = torch.tensor(dig, dtype=m.fourier_encoder.pos_encoder.Wr.weight.dtype).unsqueeze(1).to(p.device)
    emb_f = m.fourier_encoder.pos_encoder(x)
    return m.final_linear(torch.cat([emb_geo, emb_f], dim=-1))
