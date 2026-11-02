"""
Konuşmadan Duygu Tanıma - Standart öznitelik setleri ve önceden eğitilmiş gömmeler (H4 kapsamı)  |  Takım: Tertipler

preprocess.py çıktısını (X_*.npy) okur, features.py ile AYNI sırada ve aynı klasöre yazar:
  egemaps_{split}.npy : (N, 88)    eGeMAPSv02 functionals [11]  (openSMILE)
  compare_{split}.npy : (N, 6373)  ComParE 2016 functionals     (openSMILE)
  w2v_{split}.npy     : (N, 768)   wav2vec 2.0 [14] gömmesi (seçilen katmanın zaman ortalaması)
  w2v_layers_{split}.npy : (N, 13, 768) float16, tüm katmanların zaman ortalaması (katman analizi için)

wav2vec 2.0 modeli İNCE AYAR YAPILMADAN (dondurulmuş) kullanılır; yalnızca öznitelik çıkarıcıdır.
Katman seçimi: her katman için doğrulama (val) kümesinde lojistik regresyon makro F1'i ölçülür, en iyisi seçilir
(test kümesine bakılmaz). Orta katmanların duyguda daha başarılı olduğu literatürde bildirilmiştir.

Sondaki sıfır dolgusu (padding) openSMILE ve wav2vec ortalamasından çıkarılır; böylece sessiz dolgu
öznitelikleri bozmaz.

Kullanım (Colab, GPU önerilir):
  pip install opensmile transformers
  python src/extra_features.py --in_dir processed_v3 --out_dir features_v3 --sets egemaps compare w2v
  # çok dilli model (Türkçe testleri için ileride):  --w2v_model facebook/wav2vec2-large-xlsr-53
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed

SR = 16000
SPLITS = ["train", "val", "test"]
MIN_LEN = SR // 2


def strip_pad(y):
    """Sondaki sıfır dolgusunu kaldırır (en az 0,5 sn bırakır)."""
    nz = np.flatnonzero(y)
    end = nz[-1] + 1 if len(nz) else len(y)
    return y[:max(end, MIN_LEN)]


# ---------------- openSMILE ----------------
def _smile_chunk(chunk, set_name):
    import opensmile
    fs = {"egemaps": opensmile.FeatureSet.eGeMAPSv02, "compare": opensmile.FeatureSet.ComParE_2016}[set_name]
    smile = opensmile.Smile(feature_set=fs, feature_level=opensmile.FeatureLevel.Functionals)
    return np.stack([smile.process_signal(strip_pad(y), SR).to_numpy()[0] for y in chunk]).astype(np.float32), \
        list(smile.feature_names)


def opensmile_features(X, set_name, n_jobs):
    n_chunks = max(1, min(len(X), 64))
    chunks = np.array_split(np.arange(len(X)), n_chunks)
    res = Parallel(n_jobs=n_jobs)(delayed(_smile_chunk)(X[c], set_name) for c in chunks if len(c))
    F = np.concatenate([r[0] for r in res])
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    return F, res[0][1]


# ---------------- wav2vec 2.0 ----------------
def w2v_layers(X, model_name, batch, device):
    import torch
    from transformers import AutoFeatureExtractor, AutoModel
    fe = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device).eval()
    hop = 320                                            # wav2vec 2.0: 20 ms / çerçeve
    out = []
    t0 = time.time()
    for i in range(0, len(X), batch):
        ys = [strip_pad(y) for y in X[i:i + batch]]
        lens = [len(y) for y in ys]
        inp = fe(ys, sampling_rate=SR, return_tensors="pt", padding=True)
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            hs = model(inp["input_values"].to(device)).hidden_states        # (katman+1) x (B, T, D)
        H = torch.stack(hs, 1).float()                                       # (B, L, T, D)
        T = H.shape[2]
        mask = torch.zeros(len(ys), T, device=H.device)
        for j, n in enumerate(lens):
            mask[j, :max(1, min(T, int(np.ceil(n / hop))))] = 1             # dolgu çerçevelerini dışla
        pooled = (H * mask[:, None, :, None]).sum(2) / mask.sum(1)[:, None, None]
        out.append(pooled.cpu().numpy().astype(np.float16))
        if (i // batch) % 20 == 0:
            print(f"   w2v {i + len(ys)}/{len(X)}  ({time.time() - t0:.0f} sn)")
    return np.concatenate(out)                                               # (N, L, D)


def select_layer(Ltr, ytr, Lva, yva):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score
    scores = []
    for l in range(Ltr.shape[1]):
        sc = StandardScaler().fit(Ltr[:, l].astype(np.float32))
        clf = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced")
        clf.fit(sc.transform(Ltr[:, l].astype(np.float32)), ytr)
        p = clf.predict(sc.transform(Lva[:, l].astype(np.float32)))
        scores.append(float(f1_score(yva, p, average="macro")))
        print(f"   katman {l:2d}: val makro F1 = {scores[-1]:.3f}")
    return int(np.argmax(scores)), scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True, help="features.py çıktısıyla aynı klasör önerilir")
    ap.add_argument("--sets", nargs="+", default=["egemaps", "w2v"], choices=["egemaps", "compare", "w2v"])
    ap.add_argument("--w2v_model", default="facebook/wav2vec2-base")
    ap.add_argument("--w2v_layer", type=int, default=None, help="katmanı elle seç (verilmezse val ile seçilir)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n_jobs", type=int, default=-1)
    args = ap.parse_args()
    inp, out = Path(args.in_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    X = {s: np.load(inp / f"X_{s}.npy") for s in SPLITS}
    y = {s: np.load(inp / f"y_{s}.npy") for s in SPLITS}
    info_path = out / "extra_features_info.json"
    info = json.load(open(info_path)) if info_path.exists() else {}

    for name in [s for s in args.sets if s != "w2v"]:
        t0 = time.time()
        for s in SPLITS:
            F, cols = opensmile_features(X[s], name, args.n_jobs)
            np.save(out / f"{name}_{s}.npy", F)
            np.save(out / f"y_{s}.npy", y[s])
            print(f"{name} {s}: {F.shape}")
        info[name] = {"n_features": int(F.shape[1]), "seconds": round(time.time() - t0)}
        (out / f"{name}_feature_names.txt").write_text("\n".join(cols), encoding="utf-8")

    if "w2v" in args.sets:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("wav2vec modeli:", args.w2v_model, "| cihaz:", device)
        L = {}
        for s in SPLITS:
            L[s] = w2v_layers(X[s], args.w2v_model, args.batch, device)
            np.save(out / f"w2v_layers_{s}.npy", L[s])
            print(f"w2v_layers {s}: {L[s].shape}")
        if args.w2v_layer is None:
            best, scores = select_layer(L["train"], y["train"], L["val"], y["val"])
        else:
            best, scores = args.w2v_layer, None
        for s in SPLITS:
            np.save(out / f"w2v_{s}.npy", L[s][:, best].astype(np.float32))
            np.save(out / f"y_{s}.npy", y[s])
        info["w2v"] = {"model": args.w2v_model, "layer": best, "val_macro_f1_by_layer": scores,
                       "n_features": int(L["train"].shape[2])}
        print(f"Seçilen wav2vec katmanı: {best}")
    json.dump(info, open(info_path, "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
