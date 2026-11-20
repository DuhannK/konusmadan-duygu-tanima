"""
Konuşmadan Duygu Tanıma - Mel-spektrogram girdili 2B CNN (H4)  |  Takım: Tertipler   (rapor Bölüm 2.5)

Model girdisi ham dalga biçimidir (16 kHz, 3 sn); log-mel spektrogram (128 mel bandı, 25 ms Hamming pencere,
10 ms kaydırma -> 128 x 301) modelin İÇİNDE, features.py/librosa ile birebir aynı ayarlarla GPU'da hesaplanır.
Böylece demo uygulaması da aynı ön ucu (front-end) kullanır ve eğitim/kullanım arasında fark oluşmaz.

Mimari (--model cnn): 4 konvolüsyon bloğu [Conv3x3-BN-ReLU-Conv3x3-BN-ReLU-MaxPool-Dropout] (32-64-128-256 kanal)
        -> küresel ortalama havuzlama -> Dropout -> Linear(6)  (~1,2 M parametre)  [12, 13]
Mimari (--model cnn_lstm): aynı 4 konvolüsyon bloğu -> frekans ekseninde ortalama (256 kanal x 18 zaman adımı)
        -> çift yönlü LSTM (128 gizli birim) -> zaman ekseninde ortalama -> Dropout -> Linear(6)  [12, 13]
Eğitim: AdamW + OneCycle, sınıf ağırlıklı çapraz entropi (+ etiket yumuşatma), SpecAugment [15]
        (frekans/zaman maskeleme) ve rastgele zaman kaydırma. Normalizasyon (mel bandı başına ortalama/std)
        yalnızca eğitim verisinden hesaplanır.
Hafta 5/7 düzeltmesi: OneCycle takvimi her zaman sonuna kadar (varsayılan 40 epok) çalışır; erken durdurma
        varsayılan olarak kapalıdır (--patience 0). Hafta 4'te erken durdurma eğitimi öğrenme oranı henüz yüksekken
        kesiyordu. Tek bölümlemede doğrulama makro F1'i en yüksek epoğun ağırlıkları geri yüklenir.
Test kümesi yalnızca bir kez, en iyi doğrulama modeliyle ölçülür.

Kullanım (Colab GPU):
  python src/cnn.py --in_dir processed_v3 --dataset all --model cnn      --out_dir cnn_all
  python src/cnn.py --in_dir processed_v3 --dataset all --model cnn_lstm --out_dir cnn_lstm_all
  # Konuşmacı bazlı 5 katlı CV (cv_eval.py ile aynı katlar; sabit epok, doğrulama kümesi kullanılmaz):
  python src/cnn.py --in_dir processed_v3 --dataset all --model cnn      --cv 5 --out_dir cnn_cv_all
  python src/cnn.py --in_dir processed_v3 --dataset all --model cnn_lstm --cv 5 --out_dir cnn_lstm_cv_all
"""
import argparse, json, math, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SR, N_FFT, HOP, N_MELS, DURATION = 16000, 400, 160, 128, 3.0
CLASSES = ["angry", "happy", "sad", "fear", "disgust", "neutral"]


# ======================= Model =======================
def mel_filterbank():
    import librosa
    return torch.tensor(librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS), dtype=torch.float32)


class LogMel(nn.Module):
    """librosa.feature.melspectrogram(..., window='hamming') + power_to_db(ref=np.max, top_db=80) eşdeğeri."""

    def __init__(self):
        super().__init__()
        self.register_buffer("fb", mel_filterbank())                                  # (128, 201)
        self.register_buffer("win", torch.hamming_window(N_FFT, periodic=True))

    def forward(self, x):                                                            # x: (B, n_örnek)
        # Hafta 7 düzeltmesi: ön uç her zaman fp32'de hesaplanır. GPU'daki karma hassasiyette (autocast, fp16)
        # 1e-10 sıfıra yuvarlanıyor; tamamen sessiz bir kayıtta log10(0) = -inf, -inf - (-inf) = NaN oluyordu.
        # Bu NaN, eğitimde BatchNorm istatistiklerini bozup modeli tek sınıf tahmin etmeye itiyordu.
        with torch.autocast(device_type=x.device.type, enabled=False):
            spec = torch.stft(x.float(), N_FFT, HOP, N_FFT, self.win.float(), center=True, pad_mode="constant",
                              return_complex=True).abs() ** 2                        # (B, 201, T)
            S = torch.matmul(self.fb.float(), spec)                                  # (B, 128, T)
            db = 10 * torch.log10(S.clamp_min(1e-10))
            db = db - db.amax(dim=(1, 2), keepdim=True)
            return db.clamp_min(-80.0)


def _band_mask(B, L, max_w, dev):
    """Her örnek için [s, s+w) aralığını True yapan (B, L) maske; tamamen GPU'da, döngüsüz."""
    w = torch.randint(0, max_w + 1, (B, 1), device=dev)
    s = (torch.rand(B, 1, device=dev) * (L - w + 1)).long()
    a = torch.arange(L, device=dev)[None, :]
    return (a >= s) & (a < s + w)


def spec_augment(m, n_freq=2, F_w=15, n_time=2, T_w=30):
    """SpecAugment [15]: normalize edilmiş log-mel (B, 1, F, T) üzerinde frekans/zaman maskeleri (0 = ortalama).
    Hafta 4 hızlandırma: örnek örnek Python döngüsü yerine vektörel maskeler."""
    B, _, Fb, T = m.shape
    keep = torch.ones(B, 1, Fb, T, dtype=torch.bool, device=m.device)
    for _ in range(n_freq):
        keep &= ~_band_mask(B, Fb, F_w, m.device)[:, None, :, None]
    for _ in range(n_time):
        keep &= ~_band_mask(B, T, T_w, m.device)[:, None, None, :]
    return m * keep


def block(cin, cout, p):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.MaxPool2d(2), nn.Dropout(p))


MODEL_NAMES = {"cnn": "CNN", "cnn_lstm": "CNN-LSTM"}


class SERCNN(nn.Module):
    """model_type='cnn' -> küresel ortalama havuzlama; 'cnn_lstm' -> çift yönlü LSTM ile zaman modellemesi."""

    def __init__(self, n_classes=6, chans=(32, 64, 128, 256), drop=0.1, head_drop=0.3, specaug=True,
                 model_type="cnn", lstm_hidden=128):
        super().__init__()
        assert model_type in MODEL_NAMES, model_type
        self.model_type = model_type
        self.frontend = LogMel()
        self.register_buffer("mu", torch.zeros(N_MELS, 1))
        self.register_buffer("sd", torch.ones(N_MELS, 1))
        self.specaug = specaug
        layers, cin = [], 1
        for c in chans:
            layers.append(block(cin, c, drop)); cin = c
        self.body = nn.Sequential(*layers)
        if model_type == "cnn_lstm":
            self.lstm = nn.LSTM(cin, lstm_hidden, batch_first=True, bidirectional=True)
            cin = 2 * lstm_hidden
        self.head = nn.Sequential(nn.Dropout(head_drop), nn.Linear(cin, n_classes))

    def features(self, x):
        m = (self.frontend(x) - self.mu) / self.sd                                    # (B, 128, T)
        return m.unsqueeze(1)

    def forward(self, x):
        m = self.features(x)
        if self.training and self.specaug:
            m = spec_augment(m)
        h = self.body(m)                                                             # (B, C, F', T')
        if self.model_type == "cnn_lstm":
            seq = h.mean(dim=2).transpose(1, 2)                                      # (B, T', C): zaman dizisi
            out, _ = self.lstm(seq.float())                                          # (B, T', 2*gizli)
            h = out.mean(dim=1)                                                      # zaman ekseninde ortalama
        else:
            h = h.mean(dim=(2, 3))                                                   # küresel ortalama havuzlama
        return self.head(h)

    @torch.no_grad()
    def fit_norm(self, X, device, n=2000, bs=128):
        """Mel bandı başına ortalama/std - yalnızca eğitim örneklerinden."""
        idx = np.random.default_rng(0).choice(len(X), min(n, len(X)), replace=False)
        s, s2, cnt = 0, 0, 0
        for i in range(0, len(idx), bs):
            m = self.frontend(torch.from_numpy(X[np.sort(idx[i:i + bs])]).to(device))
            s = s + m.sum(dim=(0, 2)); s2 = s2 + (m ** 2).sum(dim=(0, 2)); cnt += m.shape[0] * m.shape[2]
        mu = s / cnt
        self.mu.copy_(mu[:, None]); self.sd.copy_((s2 / cnt - mu ** 2).clamp_min(1e-6).sqrt()[:, None])


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = SERCNN(**ck.get("arch", {})).to(device)
    model.load_state_dict(ck["state_dict"])
    return model.eval(), ck


# ======================= Eğitim =======================
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def time_shift(x, max_shift):
    """Örnek başına rastgele dairesel kaydırma (torch.roll eşdeğeri), GPU'da tek işlemde."""
    if max_shift <= 0:
        return x
    B, N = x.shape
    s = torch.randint(-max_shift, max_shift + 1, (B, 1), device=x.device)
    idx = (torch.arange(N, device=x.device)[None, :] - s) % N
    return x.gather(1, idx)


@torch.no_grad()
def predict(model, X, device, bs=128):
    model.eval()
    P = []
    for i in range(0, len(X), bs):
        xb = X[i:i + bs]
        xb = (torch.from_numpy(xb) if isinstance(xb, np.ndarray) else xb).to(device)
        with torch.autocast(device_type="cuda", enabled=device == "cuda"):
            P.append(F.softmax(model(xb).float(), 1).cpu().numpy())
    return np.concatenate(P)


@torch.no_grad()
def check_finite(model, X, device, log_prefix="", bs=256):
    """Eğitimden önce tüm eğitim örneklerinin normalize log-mel'ini (eğitimdeki karma hassasiyetle) hesaplar;
    sonlu olmayan (NaN/Inf) değer üreten örnek varsa eğitimi durdurur."""
    bad = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(device)
        with torch.autocast(device_type="cuda", enabled=device == "cuda"):
            m = model.features(xb)
        ok = torch.isfinite(m).flatten(1).all(1).cpu().numpy()
        bad += (np.where(~ok)[0] + i).tolist()
    n_silent = int(np.sum(~np.any(X != 0, axis=1)))
    print(f"{log_prefix}ön uç kontrolü: {len(X)} örnek, NaN/Inf {len(bad)}, tamamen sessiz {n_silent}")
    if bad:
        raise RuntimeError(f"Ön uç {len(bad)} örnekte NaN/Inf üretti (ilk indeksler: {bad[:10]})")


def train_model(Xtr, ytr, Xva, yva, args, device, log_prefix=""):
    """Xva None ise sabit epok (CV modu). Döner: (en iyi model, geçmiş)."""
    from sklearn.metrics import f1_score
    model = SERCNN(drop=args.dropout, head_drop=args.head_dropout, specaug=not args.no_specaug,
                   model_type=args.model).to(device)
    model.fit_norm(Xtr, device)
    check_finite(model, Xtr, device, log_prefix)
    counts = np.bincount(ytr, minlength=len(CLASSES))
    w = torch.tensor(len(ytr) / (len(CLASSES) * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=args.label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = math.ceil(len(Xtr) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    max_shift = int(args.time_shift * SR)
    # Hafta 4 hızlandırma: eğitim/doğrulama verisi bir kez GPU belleğine taşınır (her adımda CPU->GPU kopyası yok)
    Xtr_d = torch.from_numpy(Xtr).to(device)
    ytr_d = torch.from_numpy(ytr).long().to(device)
    Xva_d = torch.from_numpy(Xva).to(device) if Xva is not None else None
    best, best_state, bad, hist = -1, None, 0, []
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time()
        perm = torch.randperm(len(Xtr), device=device); tot, correct, n = 0.0, 0, 0
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            xb = time_shift(Xtr_d[b], max_shift)
            yb = ytr_d[b]
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                out = model(xb); loss = crit(out.float(), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(b); correct += (out.argmax(1) == yb).sum().item(); n += len(b)
        h = dict(epoch=ep, train_loss=tot / n, train_acc=correct / n)
        if Xva is not None:
            pv = predict(model, Xva_d, device, bs=max(128, args.batch)).argmax(1)
            h.update(val_acc=float((pv == yva).mean()), val_f1=float(f1_score(yva, pv, average="macro")))
            if h["val_f1"] > best:
                best, bad = h["val_f1"], 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
        hist.append(h)
        msg = " ".join(f"{k}={v:.3f}" for k, v in h.items() if k != "epoch")
        print(f"{log_prefix}epok {ep:3d} | {msg} | {time.time() - t0:.0f} sn")
        if Xva is not None and args.patience > 0 and bad >= args.patience:
            print(f"{log_prefix}erken durdurma (en iyi val F1 = {best:.3f})")
            break
    if best_state is not None:                                    # en iyi doğrulama epoğunun ağırlıkları
        model.load_state_dict(best_state)
        print(f"{log_prefix}en iyi doğrulama epoğu geri yüklendi (val F1 = {best:.3f})")
    return model, hist


def save_ckpt(model, args, path, extra=None):
    torch.save({"state_dict": model.state_dict(), "classes": CLASSES, "sr": SR, "duration": DURATION,
                "arch": dict(drop=args.dropout, head_drop=args.head_dropout, model_type=args.model),
                "train_args": vars(args),
                **(extra or {})}, path)


def plot_history(hist, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(ep, [h["train_loss"] for h in hist]); ax[0].set_title("Eğitim kaybı"); ax[0].set_xlabel("epok")
    ax[1].plot(ep, [h["train_acc"] for h in hist], label="eğitim doğruluk")
    if "val_acc" in hist[0]:
        ax[1].plot(ep, [h["val_acc"] for h in hist], label="doğrulama doğruluk")
        ax[1].plot(ep, [h["val_f1"] for h in hist], label="doğrulama makro F1")
    ax[1].legend(); ax[1].set_xlabel("epok"); ax[1].set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="preprocess.py çıktısı (X_*.npy, metadata.csv)")
    ap.add_argument("--dataset", default="all", choices=["all", "ravdess", "cremad"])
    ap.add_argument("--out_dir", default="cnn_results")
    ap.add_argument("--model", default="cnn", choices=list(MODEL_NAMES), help="cnn | cnn_lstm")
    ap.add_argument("--epochs", type=int, default=40, help="OneCycle takviminin tamamı bu kadar epok sürer")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--head_dropout", type=float, default=0.3)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--time_shift", type=float, default=0.25, help="rastgele zaman kaydırma (sn)")
    ap.add_argument("--no_specaug", action="store_true")
    ap.add_argument("--no_aug", action="store_true", help="preprocess'teki artırılmış kopyaları kullanma")
    ap.add_argument("--patience", type=int, default=0, help="0 = erken durdurma kapalı (önerilen)")
    ap.add_argument("--cv", type=int, default=0, help="k > 0 ise konuşmacı bazlı k katlı CV (sabit epok)")
    ap.add_argument("--demographics", help="CV için (cv_eval.py ile aynı verilmeli)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from folds import pooled_metadata, load_pooled, make_folds, fold_signature, select_dataset, check_no_leak
    from baseline import metrics, plot_cm, LABELS
    from sklearn.metrics import confusion_matrix

    set_seed(args.seed)
    tag, name = args.model, MODEL_NAMES[args.model]          # dosya adı etiketi / raporlardaki model adı
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True             # sabit girdi boyutu (128x301) -> en hızlı konvolüsyon seçilir
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    pool = pooled_metadata(args.in_dir)
    X = load_pooled(args.in_dir, "X", args.in_dir).astype(np.float32)
    y = pool.label.to_numpy().astype(np.int64)
    sel = select_dataset(pool, args.dataset)
    is_aug = pool.is_aug.to_numpy()
    dsid = (pool.dataset == "cremad").to_numpy().astype(int)
    print(f"Model: {name} | cihaz: {device} | veri: {args.dataset} | toplam {sel.sum()} kayıt (artırılmış dahil)")
    json.dump(vars(args), open(out / "args.json", "w"), indent=2)

    if args.cv:                                                                     # ---------- CV modu
        folds = make_folds(pool, args.cv, args.seed, args.demographics)
        check_no_leak(pool, folds, sel)
        eval_idx = np.where(sel & ~is_aug)[0]
        f_oof, y_oof = folds[eval_idx], y[eval_idx]
        oof = np.full(len(eval_idx), -1); fold_m = []
        for f in range(args.cv):
            set_seed(args.seed + f)
            tr = np.where(sel & (folds != f) & (~is_aug if args.no_aug else True))[0]
            te_pos = np.where(f_oof == f)[0]; te = eval_idx[te_pos]
            model, hist = train_model(X[tr], y[tr], None, None, args, device, log_prefix=f"[kat {f}] ")
            pred = predict(model, X[te], device).argmax(1)
            oof[te_pos] = pred
            m = metrics(y[te], pred); fold_m.append(m)
            print(f"[kat {f}] TEST acc={m['acc']:.3f} F1={m['macro_f1']:.3f} UAR={m['uar']:.3f}")
        np.save(out / f"oof_{tag}.npy", oof); np.save(out / "oof_index.npy", eval_idx)
        cm = confusion_matrix(y_oof, oof, labels=LABELS)
        plot_cm(cm, f"{name} - {args.cv} katlı CV", out / f"cm_oof_{tag}.png")
        agg = {k: (float(np.mean([m[k] for m in fold_m])), float(np.std([m[k] for m in fold_m])))
               for k in ["acc", "macro_f1", "uar"]}
        json.dump({"fold_signature": fold_signature(pool, folds), "dataset": args.dataset, "k": args.cv,
                   "models": {name: {"folds": fold_m, "mean_std": agg, "oof_file": f"oof_{tag}.npy",
                                      "pooled_oof": metrics(y_oof, oof)}}},
                  open(out / "cv_summary.json", "w"), indent=2, ensure_ascii=False)
        print(f"\n{name} {args.cv} katlı CV: doğruluk {agg['acc'][0]*100:.1f} ± {agg['acc'][1]*100:.1f}% | "
              f"makro F1 {agg['macro_f1'][0]:.3f} ± {agg['macro_f1'][1]:.3f}")
        return

    split = pool.split.to_numpy()                                                    # ---------- tek bölümleme
    tr = np.where(sel & (split == "train") & (~is_aug if args.no_aug else True))[0]
    va = np.where(sel & (split == "val"))[0]
    te = np.where(sel & (split == "test"))[0]
    print(f"eğitim/doğrulama/test = {len(tr)}/{len(va)}/{len(te)}")
    model, hist = train_model(X[tr], y[tr], X[va], y[va], args, device)
    plot_history(hist, out / "history.png")

    probs = predict(model, X[te], device); pred = probs.argmax(1)                   # test: bir kez
    val_m = metrics(y[va], predict(model, X[va], device).argmax(1))
    test_m = metrics(y[te], pred)
    by_ds = {nm: metrics(y[te][dsid[te] == k], pred[dsid[te] == k])
             for k, nm in {0: "RAVDESS", 1: "CREMA-D"}.items() if (dsid[te] == k).any()}
    cm = confusion_matrix(y[te], pred, labels=LABELS)
    plot_cm(cm, f"{name} - test", out / f"cm_{tag}.png")
    np.save(out / "pred_test.npy", pred); np.save(out / "probs_test.npy", probs)
    np.save(out / "y_test.npy", y[te]); np.save(out / "dataset_test.npy", dsid[te])
    n_params = sum(p.numel() for p in model.parameters())
    save_ckpt(model, args, out / f"{tag}_best.pt", {"val": val_m, "test": test_m})
    res = dict(val=val_m, test=test_m, test_by_dataset=by_ds, confusion_matrix=cm.tolist(),
               per_class_recall=dict(zip(CLASSES, (cm.diagonal() / np.maximum(cm.sum(1), 1)).tolist())),
               epochs_run=len(hist), best_epoch=int(np.argmax([h["val_f1"] for h in hist]) + 1),
               n_params=int(n_params), history=hist)
    json.dump(res, open(out / "results.json", "w"), indent=2, ensure_ascii=False)
    print(f"\n{name} TEST acc={test_m['acc']:.3f} makroF1={test_m['macro_f1']:.3f} UAR={test_m['uar']:.3f} "
          f"| en iyi epok {res['best_epoch']} | {n_params/1e6:.2f} M parametre")
    for k, v in by_ds.items():
        print(f"   {k}: acc={v['acc']:.3f} F1={v['macro_f1']:.3f}")
    ok = test_m["acc"] >= 0.65 and test_m["macro_f1"] >= 0.60
    print("Hedef (H4): doğruluk >= %65 ve makro F1 >= 0,60 ->", "ULAŞILDI" if ok else "ULAŞILAMADI")


if __name__ == "__main__":
    main()
