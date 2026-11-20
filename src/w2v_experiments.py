"""
Konuşmadan Duygu Tanıma - wav2vec 2.0 katman birleşimi deneyleri (H4/H5)  |  Takım: Tertipler

extra_features.py'nin ürettiği w2v_layers_{split}.npy (N, 13, 768) dosyalarını kullanır ve wav2vec 2.0'ın
hangi katman(lar)ının duygu için en iyi gösterimi verdiğini karşılaştırır [14, 19]:

  tek_en_iyi : extra_features.py'nin doğrulamada seçtiği tek katman (Hafta 4: katman 5)
  tum_ort    : 13 katmanın (CNN çıkışı + 12 transformer katmanı) eşit ağırlıklı ortalaması
  orta_ort   : orta katmanların eşit ağırlıklı ortalaması (varsayılan 3-9; --mid ile değiştirilebilir)
  en_iyi_k   : doğrulamada en iyi k tek katmanın ortalaması (varsayılan k=3)

Adımlar:
  1) Tek bölümleme: her strateji için StandardScaler (yalnızca eğitim) + SVM (RBF). C ve gamma doğrulama
     kümesinde makro F1'e göre seçilir. Test kümesi her strateji için bir kez ölçülür.
  2) En iyi strateji TEST'e değil DOĞRULAMA makro F1'ine göre seçilir.
  3) Seçilen strateji (ve --cv_all ile hepsi) konuşmacı bazlı 5 katlı CV ile ölçülür. Katlar folds.py ile
     cv_eval.py / cnn.py --cv ile birebir aynıdır; çıktı cv_eval.py --include_dir ile McNemar/t-testlerine eklenir.
  4) Seçilen stratejinin son modeli (scaler + SVM + katman listesi) arayüz (app.py) için kaydedilir.

Not: Katman/strateji seçimi sabit doğrulama kümesinde yapıldığından CV sonucunda küçük bir iyimserlik payı olabilir.

Kullanım:
  python src/w2v_experiments.py --in_dir processed_v3 --feat_dir features_v3 --dataset all --out_dir w2v_exp_all
  streamlit run src/app.py -- --w2v_dir w2v_exp_all/model --feat_info w2v_exp_all/model/w2v_info.json
"""
import argparse, itertools, json, os, time
from pathlib import Path
import numpy as np
import joblib
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

from folds import pooled_metadata, load_pooled, make_folds, fold_signature, select_dataset, check_no_leak
from baseline import metrics, plot_cm, CLASSES, LABELS

SPLITS = ["train", "val", "test"]
GRID = {"C": [1, 10, 100], "gamma": ["scale", 1e-3, 3e-4]}
SEED = 42
DS_NAMES = {0: "RAVDESS", 1: "CREMA-D"}


def make_svm(**p):
    return SVC(kernel="rbf", class_weight="balanced", random_state=SEED, **p)


def pool_layers(L, layers):
    """(N, katman, D) float16 -> seçilen katmanların ortalaması (N, D) float32."""
    return L[:, layers, :].astype(np.float32).mean(axis=1)


def fit_eval(params, Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    model = make_svm(**params).fit(sc.transform(Xtr), ytr)
    return model.predict(sc.transform(Xte)), sc, model


def tune(Xtr, ytr, Xva, yva, n_jobs):
    grid = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    res = Parallel(n_jobs=n_jobs)(delayed(fit_eval)(p, Xtr, ytr, Xva) for p in grid)
    scores = [metrics(yva, r[0])["macro_f1"] for r in res]
    b = int(np.argmax(scores))
    return grid[b], scores[b]


def cv_job(f, X, y, sel, folds, eval_idx, f_oof, params):
    tr = sel & (folds != f)
    te_pos = np.where(f_oof == f)[0]
    te = eval_idx[te_pos]
    pred, _, _ = fit_eval(params, X[tr], y[tr], X[te])
    return f, te_pos, pred, metrics(y[te], pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="preprocess.py çıktısı (metadata.csv)")
    ap.add_argument("--feat_dir", required=True, help="w2v_layers_*.npy ve extra_features_info.json klasörü")
    ap.add_argument("--dataset", default="all", choices=["all", "ravdess", "cremad"])
    ap.add_argument("--mid", nargs=2, type=int, default=[3, 9], help="orta katman aralığı (dahil)")
    ap.add_argument("--k", type=int, default=3, help="en_iyi_k stratejisindeki katman sayısı")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cv_all", action="store_true", help="tüm stratejileri CV ile ölç (varsayılan: yalnızca seçileni)")
    ap.add_argument("--demographics", help="cv_eval.py ile aynı verilmeli")
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out_dir", default="w2v_exp")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    n_jobs = args.n_jobs if args.n_jobs > 0 else (os.cpu_count() or 1)
    fd = Path(args.feat_dir)

    # ---------------- veri (havuz sırası: train, val, test) ----------------
    pool = pooled_metadata(args.in_dir)
    Ls = {s: np.load(fd / f"w2v_layers_{s}.npy", mmap_mode="r") for s in SPLITS}
    L = np.concatenate([Ls[s] for s in SPLITS])
    assert len(L) == len(pool), f"w2v_layers satır sayısı ({len(L)}) metadata ({len(pool)}) ile uyuşmuyor"
    y = pool.label.to_numpy()
    split = pool.split.to_numpy()
    dsid = (pool.dataset == "cremad").to_numpy().astype(int)
    sel = select_dataset(pool, args.dataset)
    is_aug = pool.is_aug.to_numpy()
    n_layers = L.shape[1]
    info_path = fd / "extra_features_info.json"
    info = json.load(open(info_path)) if info_path.exists() else {}
    w2v_info = info.get("w2v", {})
    print(f"wav2vec katmanları: {L.shape} | veri: {args.dataset} | çekirdek: {n_jobs}")

    tr = np.where(sel & (split == "train"))[0]           # artırılmış kopyalar eğitimde
    va = np.where(sel & (split == "val"))[0]
    te = np.where(sel & (split == "test"))[0]

    # katman başına doğrulama skoru (en_iyi_k için ve rapor grafiği için) - hızlı: sabit SVM(C=10, scale)
    print("Katman başına doğrulama makro F1 (SVM C=10):")
    Ltr, Lva = L[tr], L[va]
    preds = Parallel(n_jobs=n_jobs)(delayed(fit_eval)({"C": 10, "gamma": "scale"}, pool_layers(Ltr, [l]), y[tr],
                                                      pool_layers(Lva, [l])) for l in range(n_layers))
    layer_f1 = [float(metrics(y[va], r[0])["macro_f1"]) for r in preds]
    del preds
    for l, f1 in enumerate(layer_f1):
        print(f"   katman {l:2d}: {f1:.3f}")

    best_single = int(w2v_info.get("layer", int(np.argmax(layer_f1))))
    lo, hi = args.mid
    topk = sorted(np.argsort(layer_f1)[::-1][:args.k].tolist())
    strategies = {
        "tek_en_iyi": [best_single],
        "tum_ort": list(range(n_layers)),
        "orta_ort": list(range(lo, hi + 1)),
        f"en_iyi_{args.k}": topk,
    }

    # ---------------- 1) tek bölümleme ----------------
    results = {}
    for name, layers in strategies.items():
        t0 = time.time()
        Xtr, Xva, Xte = (pool_layers(L[i], layers) for i in (tr, va, te))
        params, val_f1 = tune(Xtr, y[tr], Xva, y[va], n_jobs)
        pred, sc, model = fit_eval(params, Xtr, y[tr], Xte)
        tm = metrics(y[te], pred)
        by_ds = {DS_NAMES[k]: metrics(y[te][dsid[te] == k], pred[dsid[te] == k])
                 for k in DS_NAMES if (dsid[te] == k).any()}
        results[name] = dict(layers=layers, params=params, val_macro_f1=val_f1, test=tm, test_by_dataset=by_ds)
        np.save(out / f"pred_test_{name}.npy", pred)
        print(f"{name:12s} katmanlar={layers} | VAL F1={val_f1:.3f} | TEST acc={tm['acc']:.3f} "
              f"F1={tm['macro_f1']:.3f} UAR={tm['uar']:.3f} | {params} ({time.time()-t0:.0f} sn)")
    np.save(out / "y_test.npy", y[te]); np.save(out / "dataset_test.npy", dsid[te])

    chosen = max(results, key=lambda n: results[n]["val_macro_f1"])      # TEST'e bakılmadan seçim
    print(f"\nDoğrulamaya göre seçilen strateji: {chosen} {strategies[chosen]}")

    # ---------------- 4) arayüz için son model ----------------
    Xtr = pool_layers(L[tr], strategies[chosen])
    sc = StandardScaler().fit(Xtr)
    final = make_svm(**results[chosen]["params"]).fit(sc.transform(Xtr), y[tr])
    md = out / "model"; md.mkdir(exist_ok=True)
    joblib.dump(sc, md / "scaler.joblib"); joblib.dump(final, md / "model_svm.joblib")
    json.dump({"w2v": {"model": w2v_info.get("model", "facebook/wav2vec2-base"),
                       "layer": strategies[chosen][0], "layers": strategies[chosen], "strategy": chosen}},
              open(md / "w2v_info.json", "w"), indent=2)

    # ---------------- 3) konuşmacı bazlı k-katlı CV ----------------
    folds = make_folds(pool, args.folds, args.seed, args.demographics)
    check_no_leak(pool, folds, sel)
    eval_idx = np.where(sel & ~is_aug)[0]
    f_oof, y_oof = folds[eval_idx], y[eval_idx]
    cv_models = {}
    for name in (strategies if args.cv_all else [chosen]):
        t0 = time.time()
        X = pool_layers(L, strategies[name])
        label = f"SVM-w2v ({name})"
        jobs = Parallel(n_jobs=min(n_jobs, args.folds))(
            delayed(cv_job)(f, X, y, sel, folds, eval_idx, f_oof, results[name]["params"])
            for f in range(args.folds))
        oof = np.full(len(eval_idx), -1); fold_m = [None] * args.folds
        for f, te_pos, pred, m in jobs:
            oof[te_pos] = pred; fold_m[f] = m
        agg = {k: (float(np.mean([m[k] for m in fold_m])), float(np.std([m[k] for m in fold_m])))
               for k in ["acc", "macro_f1", "uar"]}
        tag = f"w2v_{name}"
        np.save(out / f"oof_{tag}.npy", oof)
        cm = confusion_matrix(y_oof, oof, labels=LABELS)
        plot_cm(cm, f"{label} - {args.folds} katlı CV", out / f"cm_oof_{tag}.png")
        cv_models[label] = dict(folds=fold_m, mean_std=agg, oof_file=f"oof_{tag}.npy",
                                pooled_oof=metrics(y_oof, oof), layers=strategies[name])
        results[name]["cv"] = agg
        print(f"CV {name:12s}: doğruluk {agg['acc'][0]*100:.1f} ± {agg['acc'][1]*100:.1f}% | "
              f"makro F1 {agg['macro_f1'][0]:.3f} ± {agg['macro_f1'][1]:.3f} ({time.time()-t0:.0f} sn)")
    np.save(out / "oof_index.npy", eval_idx)
    json.dump({"fold_signature": fold_signature(pool, folds), "dataset": args.dataset, "k": args.folds,
               "models": cv_models}, open(out / "cv_summary.json", "w"), indent=2, ensure_ascii=False)
    json.dump({"dataset": args.dataset, "layer_val_f1": layer_f1, "strategies": results, "chosen": chosen},
              open(out / "results.json", "w"), indent=2, ensure_ascii=False)

    # ---------------- tablo ----------------
    T = [f"# wav2vec 2.0 katman birleşimi - veri: {args.dataset}\n",
         "| Strateji | Katmanlar | Val makro F1 | Test doğruluk | Test makro F1 | CV doğruluk |",
         "|---|---|---|---|---|---|"]
    for n, r in results.items():
        cv = f"{r['cv']['acc'][0]*100:.1f} ± {r['cv']['acc'][1]*100:.1f}%" if "cv" in r else "-"
        mark = " ✓" if n == chosen else ""
        T.append(f"| {n}{mark} | {r['layers']} | {r['val_macro_f1']:.3f} | {r['test']['acc']*100:.1f}% | "
                 f"{r['test']['macro_f1']:.3f} | {cv} |")
    T.append(f"\n✓ = doğrulama makro F1'ine göre seçilen strateji. Arayüz modeli: `{md}`")
    (out / "w2v_table.md").write_text("\n".join(T), encoding="utf-8")
    print("\n" + "\n".join(T))


if __name__ == "__main__":
    main()
