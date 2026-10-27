"""
Konuşmadan Duygu Tanıma - Klasik (baseline) modeller  |  Takım: Tertipler
features.py çıktısındaki feat_*.npy vektörleriyle SVM, Random Forest, k-NN ve MLP eğitir.

Kurallar (rapor Bölüm 2.5 / 2.7):
  - StandardScaler SADECE eğitim verisine uydurulur (sızıntı yok)
  - Hiperparametreler doğrulama (val) kümesinde makro F1'e göre seçilir
  - Test kümesi yalnızca en sonda, seçilen modeller için bir kez ölçülür
  - Metrikler: doğruluk, makro F1, UAR (= makro duyarlılık), karışıklık matrisi

Kullanım:
    python baseline.py --feat_dir /content/features --out_dir /content/results
"""
import argparse, json, itertools
from pathlib import Path
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = ["angry", "happy", "sad", "fear", "disgust", "neutral"]
LABELS = list(range(len(CLASSES)))
SEED = 42


def load(feat_dir):
    d = Path(feat_dir)
    out = {}
    for s in ["train", "val", "test"]:
        out[s] = (np.load(d / f"feat_{s}.npy"), np.load(d / f"y_{s}.npy"))
    ds = {s: (np.load(d / f"dataset_{s}.npy") if (d / f"dataset_{s}.npy").exists() else None)
          for s in ["train", "val", "test"]}
    return out, ds


def metrics(y, p):
    return dict(acc=accuracy_score(y, p),
                macro_f1=f1_score(y, p, average="macro", labels=LABELS, zero_division=0),
                uar=recall_score(y, p, average="macro", labels=LABELS, zero_division=0))


def candidates():
    """Her model için (ad, {parametre ızgarası}, kurucu)."""
    return {
        "SVM (RBF)": ({"C": [1, 10, 100], "gamma": [1e-4, 3e-4, 1e-3, 3e-3]},
                      lambda **k: SVC(kernel="rbf", class_weight="balanced", random_state=SEED, **k)),
        "Random Forest": ({"n_estimators": [300], "max_depth": [None, 20]},
                          lambda **k: RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=SEED, **k)),
        "k-NN": ({"n_neighbors": [3, 5, 9, 15]},
                 lambda **k: KNeighborsClassifier(**k)),
        "MLP": ({"hidden_layer_sizes": [(256,), (256, 128)], "alpha": [1e-4, 1e-2]},
                lambda **k: MLPClassifier(max_iter=400, early_stopping=True, random_state=SEED, **k)),
    }


def plot_cm(cm, title, path):
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cmn, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(LABELS); ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticks(LABELS); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Tahmin"); ax.set_ylabel("Gerçek"); ax.set_title(title)
    for i in LABELS:
        for j in LABELS:
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, fraction=0.046); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", required=True)
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--ablate_extras", action="store_true",
                    help="v2'nin son 4 ek özniteliğini ATARAK çalıştır (v1 ile adil karşılaştırma)")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    D, DS = load(args.feat_dir)
    DS_NAMES = {0: "RAVDESS", 1: "CREMA-D"}
    if args.ablate_extras:
        D = {s: (X[:, :-4], y) for s, (X, y) in D.items()}
    print("Öznitelik sayısı:", D["train"][0].shape[1])
    scaler = StandardScaler().fit(D["train"][0])             # yalnızca train!
    Xtr, ytr = scaler.transform(D["train"][0]), D["train"][1]
    Xva, yva = scaler.transform(D["val"][0]), D["val"][1]
    Xte, yte = scaler.transform(D["test"][0]), D["test"][1]
    joblib.dump(scaler, out / "scaler.joblib")

    results = {}
    for name, (grid, make) in candidates().items():
        best = None
        for vals in itertools.product(*grid.values()):
            params = dict(zip(grid.keys(), vals))
            model = make(**params).fit(Xtr, ytr)
            m = metrics(yva, model.predict(Xva))
            if best is None or m["macro_f1"] > best[0]["macro_f1"]:
                best = (m, params, model)
        val_m, params, model = best
        pred = model.predict(Xte)                              # test: bir kez
        test_m = metrics(yte, pred)
        by_ds = {}
        if DS["test"] is not None:
            for k in np.unique(DS["test"]):
                msk = DS["test"] == k
                by_ds[DS_NAMES[int(k)]] = metrics(yte[msk], pred[msk])
        cm = confusion_matrix(yte, pred, labels=LABELS)
        tag = name.split()[0].lower().replace("-", "")
        plot_cm(cm, f"{name} - test", out / f"cm_{tag}.png")
        joblib.dump(model, out / f"model_{tag}.joblib")
        results[name] = dict(params={k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()},
                             val=val_m, test=test_m, test_by_dataset=by_ds, confusion_matrix=cm.tolist())
        print(f"{name:14s} VAL acc={val_m['acc']:.3f} F1={val_m['macro_f1']:.3f} | TEST acc={test_m['acc']:.3f} "
              f"macroF1={test_m['macro_f1']:.3f} UAR={test_m['uar']:.3f} | {params}")
        if by_ds:
            print("   test veri seti bazında:", {k: f"acc={v['acc']:.3f} F1={v['macro_f1']:.3f}" for k, v in by_ds.items()})

    json.dump(results, open(out / "results.json", "w"), indent=2, ensure_ascii=False)
    names = sorted({k for r in results.values() for k in r["test_by_dataset"]})
    print("\n| Model | Test doğruluk | Makro F1 | UAR |" + "".join(f" {n} doğruluk |" for n in names))
    print("|---|---|---|---|" + "---|" * len(names))
    for n, r in results.items():
        t = r["test"]
        extra = "".join(f" {r['test_by_dataset'][k]['acc']*100:.1f}% |" for k in names)
        print(f"| {n} | {t['acc']*100:.1f}% | {t['macro_f1']:.3f} | {t['uar']:.3f} |" + extra)
    print("\nHedef (H3): test doğruluğu >= %55 ->",
          "ULAŞILDI" if max(r["test"]["acc"] for r in results.values()) >= 0.55 else "ULAŞILAMADI")


if __name__ == "__main__":
    main()
