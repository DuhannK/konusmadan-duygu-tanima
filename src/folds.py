"""
Konuşmadan Duygu Tanıma - Konuşmacı bazlı k-katlı bölümleme (ortak modül)  |  Takım: Tertipler

preprocess.py çıktısındaki train/val/test kümelerini (metadata.csv ile) tek havuzda birleştirir ve
her KONUŞMACIYI tek bir kata atar. Aynı oyuncu hiçbir zaman hem eğitim hem test katında yer almaz.

  - Her veri seti (RAVDESS, CREMA-D) ayrı ayrı katlara dağıtılır -> her katta iki veri setinden de
    konuşmacı bulunur.
  - RAVDESS'te cinsiyet oyuncu numarasından bilinir (tek = erkek); erkek ve kadınlar ayrı ayrı sırayla
    katlara dağıtılarak katlar cinsiyet dengeli tutulur. CREMA-D için --demographics verilirse aynısı yapılır.
  - Veri artırma (augmentation) kopyaları, ait oldukları konuşmacının katına bağlıdır; yalnızca EĞİTİMDE
    kullanılır, değerlendirmeye asla girmez.

cv_eval.py (klasik modeller) ve cnn.py --cv (CNN) aynı fonksiyonu aynı tohumla (seed) çağırdığı için
katlar birebir aynıdır; bu sayede modeller eşleştirilmiş testlerle (McNemar, t-testi) karşılaştırılabilir.
"""
import hashlib
import random
from pathlib import Path
import numpy as np
import pandas as pd

SPLITS = ["train", "val", "test"]
DS_ID = {"ravdess": 0, "cremad": 1}


def pooled_metadata(in_dir):
    """metadata.csv'yi X_train, X_val, X_test sırasıyla birleştirilmiş (havuz) sıraya dizer."""
    mdf = pd.read_csv(Path(in_dir) / "metadata.csv")
    parts = [mdf[mdf.split == s] for s in SPLITS]
    pool = pd.concat(parts, ignore_index=True)
    pool["is_aug"] = pool["augmentation"].astype(str) != "none"
    pool["speaker"] = pool["dataset"].astype(str) + "_" + pool["actor"].astype(str)
    return pool


def load_pooled(arr_dir, prefix, in_dir=None):
    """{prefix}_train/val/test.npy dosyalarını havuz sırasında birleştirir (metadata ile aynı sıra)."""
    d = Path(arr_dir)
    arrs = [np.load(d / f"{prefix}_{s}.npy") for s in SPLITS]
    out = np.concatenate(arrs, axis=0)
    if in_dir is not None:
        n_meta = len(pooled_metadata(in_dir))
        assert len(out) == n_meta, f"{prefix}: {len(out)} satır, metadata {n_meta} satır - sıra uyuşmuyor!"
    return out


def _gender_map(csv_path):
    if not csv_path:
        return None
    try:
        df = pd.read_csv(csv_path)
        cols = {c.lower(): c for c in df.columns}
        a = next(cols[c] for c in cols if "actor" in c)
        s = next(cols[c] for c in cols if c in ("sex", "gender"))
        return {int(r[a]): ("M" if str(r[s]).strip().lower().startswith("m") else "F") for _, r in df.iterrows()}
    except Exception as e:
        print("UYARI: demografi dosyası okunamadı:", e)
        return None


def make_folds(pool, k=5, seed=42, demographics=None):
    """Her satır için kat numarası (0..k-1) döndürür. Kat, konuşmacıya göre belirlenir."""
    rng = random.Random(seed)
    gmap = _gender_map(demographics)
    spk_fold = {}
    for ds in sorted(pool.dataset.unique()):
        actors = sorted(pool[pool.dataset == ds].actor.unique())
        if ds == "ravdess":
            groups = [[a for a in actors if a % 2 == 1], [a for a in actors if a % 2 == 0]]
        elif gmap and all(a in gmap for a in actors):
            groups = [[a for a in actors if gmap[a] == "M"], [a for a in actors if gmap[a] == "F"]]
        else:
            groups = [actors]
        offset = 0
        for g in groups:
            g = list(g); rng.shuffle(g)
            for i, a in enumerate(g):
                spk_fold[f"{ds}_{a}"] = (i + offset) % k
            offset += len(g)          # kadınlar, erkeklerin bıraktığı kattan devam eder -> kat boyları dengeli
    return pool["speaker"].map(spk_fold).to_numpy()


def fold_signature(pool, folds):
    """Kat atamasının kısa özeti: farklı betiklerde aynı katların kullanıldığını doğrulamak için."""
    key = ",".join(f"{s}:{f}" for s, f in sorted(set(zip(pool.speaker, folds))))
    return hashlib.md5(key.encode()).hexdigest()[:10]


def select_dataset(pool, which):
    """'all' | 'ravdess' | 'cremad' -> boolean maske."""
    if which == "all":
        return np.ones(len(pool), dtype=bool)
    return (pool.dataset == which).to_numpy()


def check_no_leak(pool, folds, mask):
    for f in np.unique(folds[mask]):
        tr = set(pool.speaker[mask & (folds != f)])
        te = set(pool.speaker[mask & (folds == f)])
        assert not (tr & te), f"Kat {f}: konuşmacı sızıntısı!"
