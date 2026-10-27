"""
Konuşmadan Duygu Tanıma - Öznitelik çıkarımı v3  |  Takım: Tertipler
preprocess.py çıktısını okur ve iki tür öznitelik üretir:

  1) logmel_{split}.npy : (N, 128, 301)  log-mel spektrogram  -> CNN girdisi
  2) feat_{split}.npy   : (N, 556)       istatistiksel vektör -> SVM / RF / k-NN / MLP girdisi
     552 = MFCC 40 + delta + delta-delta, ZCR, RMS, centroid, bandwidth, rolloff, chroma, F0
           (her biri için ortalama, std, min, max)
     +4  = v2 ekleri (preprocess extras_*.npy): konuşma süresi, RMS dB, tepe dB, sessiz olmayan oran

Pencere 25 ms, kaydırma 10 ms (rapor Bölüm 2.4).
Kullanım: python features.py --in_dir /content/processed_v3 --out_dir /content/features_v3 --skip_logmel
--skip_logmel: CNN kullanılmadığı için log-mel üretimini atlar (süre ve bellek kazandırır).
"""
import argparse, json
from pathlib import Path
import numpy as np
import librosa
from joblib import Parallel, delayed

SR = 16000
N_FFT = 400        # 25 ms
HOP = 160          # 10 ms
N_MELS = 128
N_MFCC = 40
FMIN, FMAX = 65, 400   # konuşma için F0 aralığı (Hz)


def stats(M):
    """(özellik, zaman) -> her özellik için [ortalama, std, min, max] düz vektör."""
    M = np.atleast_2d(M)
    return np.concatenate([M.mean(1), M.std(1), M.min(1), M.max(1)])


def logmel(y):
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                       win_length=N_FFT, window="hamming", n_mels=N_MELS)
    return librosa.power_to_db(S, ref=np.max).astype(np.float32)   # (128, 301)


def handcrafted(y):
    kw = dict(n_fft=N_FFT, hop_length=HOP, win_length=N_FFT, window="hamming")
    S = np.abs(librosa.stft(y, **kw))
    mel = librosa.feature.melspectrogram(S=S**2, sr=SR, n_mels=N_MELS)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=N_MFCC)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)

    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP)
    rms = librosa.feature.rms(S=S, frame_length=N_FFT, hop_length=HOP)
    cent = librosa.feature.spectral_centroid(S=S, sr=SR)
    bw = librosa.feature.spectral_bandwidth(S=S, sr=SR)
    roll = librosa.feature.spectral_rolloff(S=S, sr=SR)
    chroma = librosa.feature.chroma_stft(S=S**2, sr=SR)

    # F0 (perde): hızlı YIN; sadece enerjisi yeterli (konuşulan) çerçevelerde istatistik al
    f0 = librosa.yin(y, fmin=FMIN, fmax=FMAX, sr=SR, frame_length=1024, hop_length=HOP)
    n = min(len(f0), rms.shape[1])
    f0, r = f0[:n], rms[0, :n]
    voiced = f0[r > 0.1 * r.max()] if r.max() > 0 else f0
    if len(voiced) == 0:
        voiced = f0
    f0_stats = np.array([voiced.mean(), voiced.std(), voiced.min(), voiced.max()])

    vec = np.concatenate([stats(mfcc), stats(d1), stats(d2),
                          stats(zcr), stats(rms), stats(cent), stats(bw), stats(roll),
                          stats(chroma), f0_stats])
    return vec.astype(np.float32)


def one(y, skip_lm=False):
    return (None if skip_lm else logmel(y)), handcrafted(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", default="features")
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--skip_logmel", action="store_true")
    args = ap.parse_args()
    inp, out = Path(args.in_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    info = {}
    for split in ["train", "val", "test"]:
        X = np.load(inp / f"X_{split}.npy")
        res = Parallel(n_jobs=args.n_jobs)(delayed(one)(y, args.skip_logmel) for y in X)
        F = np.stack([r[1] for r in res])
        LM = None if args.skip_logmel else np.stack([r[0] for r in res])
        ex_path = inp / f"extras_{split}.npy"
        if ex_path.exists():                       # v2: ek 4 öznitelik
            F = np.concatenate([F, np.load(ex_path).astype(np.float32)], axis=1)
        assert np.isfinite(F).all() and (LM is None or np.isfinite(LM).all()), f"{split}: NaN/Inf var!"
        if LM is not None:
            np.save(out / f"logmel_{split}.npy", LM)
        np.save(out / f"feat_{split}.npy", F)
        # etiketleri de yanına kopyala (tek klasörden çalışmak için)
        np.save(out / f"y_{split}.npy", np.load(inp / f"y_{split}.npy"))
        if (inp / f"dataset_{split}.npy").exists():
            np.save(out / f"dataset_{split}.npy", np.load(inp / f"dataset_{split}.npy"))
        info[split] = {"n": int(len(X)), "feat_shape": list(F.shape)}
        if LM is not None:
            info[split]["logmel_shape"] = list(LM.shape)
        print(split, info[split])
    json.dump(info, open(out / "features_info.json", "w"), indent=2)


if __name__ == "__main__":
    main()
