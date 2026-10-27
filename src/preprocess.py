"""
Konuşmadan Duygu Tanıma - Ön işleme v3 (RAVDESS + CREMA-D)  |  Takım: Tertipler

Her veri seti kendi içinde KONUŞMACIDAN BAĞIMSIZ train/val/test'e bölünür (aynı oranlar),
sonra birleştirilir. RAVDESS bölümlemesi v2 ile birebir aynıdır (sonuçlar karşılaştırılabilir).
Çıktılar: X_*.npy, y_*.npy, extras_*.npy, dataset_*.npy (0=RAVDESS, 1=CREMA-D), metadata.csv, summary.json

Kullanım:
  python preprocess.py --ravdess_dir RAVDESS_KLASORU --cremad_dir /content/cremad/data/AudioWAV \
      --out_dir /content/processed_v3 --augment [--demographics VideoDemographics.csv]
"""
import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
from joblib import Parallel, delayed

SR = 16000
DURATION = 3.0
N_SAMPLES = int(SR * DURATION)
TRIM_DB = 30
EPS = 1e-8

CLASSES = ["angry", "happy", "sad", "fear", "disgust", "neutral"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
RAVDESS_EMOTION = {"01": "neutral", "02": "neutral",  # calm -> neutral
                   "03": "happy", "04": "sad", "05": "angry",
                   "06": "fear", "07": "disgust"}      # 08 surprised -> çıkarılır
CREMA_EMOTION = {"ANG": "angry", "HAP": "happy", "SAD": "sad",
                 "FEA": "fear", "DIS": "disgust", "NEU": "neutral"}
DS_ID = {"ravdess": 0, "cremad": 1}


def parse_ravdess(path: Path):
    parts = path.stem.split("-")
    if len(parts) != 7:
        return None
    modality, vocal, emo, inten, stmt, rep, actor = parts
    if vocal != "01" or emo not in RAVDESS_EMOTION:
        return None
    a = int(actor)
    return dict(file=str(path), actor=a, emotion=RAVDESS_EMOTION[emo], dataset="ravdess",
                gender="M" if a % 2 == 1 else "F")


def parse_cremad(path: Path):
    parts = path.stem.split("_")                        # 1001_DFA_ANG_XX
    if len(parts) != 4 or parts[2] not in CREMA_EMOTION:
        return None
    return dict(file=str(path), actor=int(parts[0]), emotion=CREMA_EMOTION[parts[2]],
                dataset="cremad", gender=None)


def load_gender(csv_path):
    """VideoDemographics.csv -> {actor_id: 'M'/'F'}. Başarısız olursa None döner."""
    try:
        df = pd.read_csv(csv_path)
        cols = {c.lower(): c for c in df.columns}
        a = next(cols[c] for c in cols if "actor" in c)
        s = next(cols[c] for c in cols if c in ("sex", "gender"))
        return {int(r[a]): ("M" if str(r[s]).strip().lower().startswith("m") else "F") for _, r in df.iterrows()}
    except Exception as e:
        print("UYARI: demografi dosyası okunamadı:", e)
        return None


def fix_length(y):
    if len(y) >= N_SAMPLES:
        start = (len(y) - N_SAMPLES) // 2
        return y[start:start + N_SAMPLES]
    return np.pad(y, (0, N_SAMPLES - len(y)))


def extras_of(y_trim):
    """Normalizasyondan ÖNCE: [süre sn, RMS dB, tepe dB, sessiz olmayan oran]."""
    dur = len(y_trim) / SR
    rms_db = 20 * np.log10(np.sqrt(np.mean(y_trim ** 2)) + EPS)
    peak_db = 20 * np.log10(np.max(np.abs(y_trim)) + EPS)
    nonsil = sum(e - s for s, e in librosa.effects.split(y_trim, top_db=TRIM_DB)) / len(y_trim)
    return np.array([dur, rms_db, peak_db, nonsil], dtype=np.float32)


def load_and_clean(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=TRIM_DB)
    if len(y) == 0:
        return None, None
    extras = extras_of(y)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return fix_length(y).astype(np.float32), extras


def speaker_split_groups(groups, seed=42):
    """groups: konuşmacı listeleri (ör. [erkekler, kadınlar] ya da [hepsi]). Her grup ~%67/17/17 bölünür."""
    rng = random.Random(seed)
    tr, va, te = [], [], []
    for lst in groups:
        lst = sorted(lst); rng.shuffle(lst)
        n = len(lst)
        n_test = max(1, round(n / 6)) if n >= 6 else 1
        n_val = n_test
        te += lst[:n_test]; va += lst[n_test:n_test + n_val]; tr += lst[n_test + n_val:]
    return {"train": sorted(tr), "val": sorted(va), "test": sorted(te)}


def augment(y, extras, rng):
    kind = rng.choice(["noise", "pitch", "stretch"])
    ex = extras.copy()
    if kind == "noise":
        out = y + rng.normal(0, 0.005 * max(np.max(np.abs(y)), EPS), size=y.shape)
    elif kind == "pitch":
        out = librosa.effects.pitch_shift(y, sr=SR, n_steps=float(rng.choice([-2, -1, 1, 2])))
    else:
        rate = float(rng.choice([0.9, 1.1]))
        out = librosa.effects.time_stretch(y, rate=rate)
        ex[0] = ex[0] / rate
    return fix_length(out.astype(np.float32)), ex, kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ravdess_dir")
    ap.add_argument("--cremad_dir")
    ap.add_argument("--demographics", help="CREMA-D VideoDemographics.csv (cinsiyet dengeli bölümleme için, isteğe bağlı)")
    ap.add_argument("--out_dir", default="processed")
    ap.add_argument("--augment", action="store_true", help="RAVDESS eğitim verisine artırılmış kopya ekle")
    ap.add_argument("--augment_cremad", action="store_true", help="CREMA-D eğitim verisine de ekle (yavaş)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not (args.ravdess_dir or args.cremad_dir):
        raise SystemExit("En az bir veri seti klasörü verin (--ravdess_dir / --cremad_dir).")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = []
    if args.ravdess_dir:
        rows += [r for p in sorted(Path(args.ravdess_dir).rglob("*.wav")) if (r := parse_ravdess(p))]
    if args.cremad_dir:
        rows += [r for p in sorted(Path(args.cremad_dir).rglob("*.wav")) if (r := parse_cremad(p))]
    if not rows:
        raise SystemExit("Uygun .wav dosyası bulunamadı. Klasör yollarını kontrol edin.")
    df = pd.DataFrame(rows)

    gender_map = load_gender(args.demographics) if args.demographics else None
    split_of, split_info = {}, {}
    for ds in df.dataset.unique():
        actors = sorted(df[df.dataset == ds].actor.unique())
        if ds == "cremad":
            if gender_map and all(a in gender_map for a in actors):
                groups = [[a for a in actors if gender_map[a] == "M"], [a for a in actors if gender_map[a] == "F"]]
            else:
                print("UYARI: CREMA-D cinsiyet bilgisi yok -> konuşmacılar cinsiyet dengesi gözetilmeden rastgele bölünür.")
                groups = [actors]
        else:
            groups = [[a for a in actors if a % 2 == 1], [a for a in actors if a % 2 == 0]]
        sp = speaker_split_groups(groups, args.seed)
        split_info[ds] = {s: [int(a) for a in lst] for s, lst in sp.items()}
        for s, lst in sp.items():
            for a in lst:
                split_of[(ds, a)] = s
    df["split"] = [split_of[(d, a)] for d, a in zip(df.dataset, df.actor)]

    print(f"{len(df)} dosya yükleniyor ve temizleniyor...")
    loaded = Parallel(n_jobs=-1)(delayed(load_and_clean)(f) for f in df.file)

    rng = np.random.default_rng(args.seed)
    data = {s: [] for s in ["train", "val", "test"]}
    meta, skipped = [], 0
    for (_, r), (y, ex) in zip(df.iterrows(), loaded):
        if y is None:
            skipped += 1
            continue
        lab, dsid = CLASS_TO_ID[r.emotion], DS_ID[r.dataset]
        data[r.split].append((y, lab, ex, dsid))
        meta.append(dict(file=Path(r.file).name, dataset=r.dataset, actor=r.actor, emotion=r.emotion,
                         label=lab, split=r.split, augmentation="none"))
        do_aug = (r.dataset == "ravdess" and args.augment) or (r.dataset == "cremad" and args.augment_cremad)
        if do_aug and r.split == "train":
            ya, exa, kind = augment(y, ex, rng)
            data["train"].append((ya, lab, exa, dsid))
            meta.append(dict(file=Path(r.file).name, dataset=r.dataset, actor=r.actor, emotion=r.emotion,
                             label=lab, split="train", augmentation=kind))

    for s, items in data.items():
        np.save(out / f"X_{s}.npy", np.stack([i[0] for i in items]))
        np.save(out / f"y_{s}.npy", np.array([i[1] for i in items]))
        np.save(out / f"extras_{s}.npy", np.stack([i[2] for i in items]))
        np.save(out / f"dataset_{s}.npy", np.array([i[3] for i in items]))

    mdf = pd.DataFrame(meta)
    mdf.to_csv(out / "metadata.csv", index=False)
    summary = {
        "sr": SR, "duration_s": DURATION, "classes": CLASSES, "skipped_empty": skipped,
        "extras": ["dur_s", "rms_db", "peak_db", "nonsilent_ratio"],
        "speakers": split_info,
        "totals": {s: int((mdf.split == s).sum()) for s in data},
        "totals_by_dataset": {ds: {s: int(((mdf.split == s) & (mdf.dataset == ds)).sum()) for s in data}
                              for ds in mdf.dataset.unique()},
        "counts": {s: mdf[mdf.split == s].emotion.value_counts().reindex(CLASSES, fill_value=0).to_dict() for s in data},
    }
    sets = {s: set(zip(mdf[mdf.split == s].dataset, mdf[mdf.split == s].actor)) for s in data}
    assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]), "Konuşmacı sızıntısı!"
    json.dump(summary, open(out / "summary.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps({k: summary[k] for k in ["totals", "totals_by_dataset", "counts"]}, indent=2, ensure_ascii=False))
    print({ds: {s: len(v) for s, v in sp.items()} for ds, sp in split_info.items()}, "<- konuşmacı sayıları")


if __name__ == "__main__":
    main()
