"""
Konuşmadan Duygu Tanıma - Konuşmacı bazlı k-katlı çapraz doğrulama (H5)  |  Takım: Tertipler

Klasik modelleri (SVM, RF, k-NN, MLP) konuşmacı bazlı k=5 katlı çapraz doğrulamayla ölçer ve
modelleri istatistiksel olarak karşılaştırır (rapor Bölüm 2.7):
  - Her katta: StandardScaler yalnızca eğitim katına uydurulur; test katında yalnızca ORİJİNAL kayıtlar ölçülür
  - Sonuçlar ortalama ± standart sapma (doğruluk, makro F1, UAR)
  - Katlar dışı (out-of-fold) tahminlerle McNemar testi (+ Holm düzeltmesi)
  - Kat bazlı skorlarla eşleştirilmiş t-testi ve Nadeau-Bengio düzeltmeli t-testi
  - --include_dir ile cnn.py --cv çıktısı (CNN) da aynı testlere dahil edilir (katlar birebir aynıdır)

Öznitelik seti --feat_name ile seçilir: feat (556, Hafta 3), egemaps (88), compare (6373), w2v (768)

Kullanım:
  python src/cv_eval.py --in_dir processed_v3 --feat_dir features_v3 --dataset ravdess \
      --params results_v3/results.json --out_dir cv_ravdess_feat
  python src/cv_eval.py ... --tune          # iç içe (nested) seçim: her katta hiperparametreler ayrıca seçilir
"""
import argparse, itertools, json, os, re, time
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from folds import pooled_metadata, load_pooled, make_folds, fold_signature, select_dataset, check_no_leak
from baseline import candidates, metrics, plot_cm, CLASSES, LABELS
from stats_tests import pairwise_mcnemar, paired_t, holm, mcnemar_table, fmt_p, ALPHA

TAGS = {"SVM (RBF)": "svm", "Random Forest": "rf", "k-NN": "knn", "MLP": "mlp"}
DEFAULT_PARAMS = {"SVM (RBF)": {"C": 10, "gamma": "scale"},
                  "Random Forest": {"n_estimators": 300, "max_depth": None},
                  "k-NN": {"n_neighbors": 9},
                  "MLP": {"hidden_layer_sizes": (256,), "alpha": 1e-2}}
DS_NAMES = {"ravdess": "RAVDESS", "cremad": "CREMA-D"}


def fixed_params(path):
    """Hafta 3 baseline.py results.json'daki doğrulama ile seçilmiş parametreler (yoksa varsayılanlar)."""
    P = {k: dict(v) for k, v in DEFAULT_PARAMS.items()}
    if path and Path(path).exists():
        res = json.load(open(path))
        for name, r in res.items():
            if name in P:
                P[name] = {k: (tuple(v) if isinstance(v, list) else v) for k, v in r["params"].items()}
        print("Parametreler", path, "dosyasından alındı.")
    return P


def tune_inner(name, Xtr, ytr, Xva, yva):
    grid, make = candidates()[name]
    best = None
    for vals in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), vals))
        m = metrics(yva, make(**params).fit(Xtr, ytr).predict(Xva))
        if best is None or m["macro_f1"] > best[0]:
            best = (m["macro_f1"], params)
    return best[1]


def run_job(n, f, X, y, sel, folds, is_aug, eval_idx, f_oof, params, tune, no_aug, k):
    """Tek bir (model, kat) çifti: ölçekle, (isteğe bağlı iç seçim), eğit, test katında tahmin et."""
    t0 = time.time()
    tr = sel & (folds != f) & (~is_aug if no_aug else True)
    te_pos = np.where(f_oof == f)[0]
    te = eval_idx[te_pos]
    sc = StandardScaler().fit(X[tr])
    Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    if tune:                                           # iç doğrulama: bir sonraki kat (eğitimden ayrılır)
        fv = (f + 1) % k
        itr = tr & (folds != fv)
        iva = sel & (folds == fv) & ~is_aug
        isc = StandardScaler().fit(X[itr])
        params = tune_inner(n, isc.transform(X[itr]), y[itr], isc.transform(X[iva]), y[iva])
    _, make = candidates()[n]
    pred = make(**params).fit(Xtr, y[tr]).predict(Xte)
    return te_pos, pred, metrics(y[te], pred), params, len(te) / tr.sum(), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="preprocess.py çıktısı (metadata.csv için)")
    ap.add_argument("--feat_dir", required=True)
    ap.add_argument("--feat_name", default="feat", help="feat | egemaps | compare | w2v")
    ap.add_argument("--dataset", default="all", choices=["all", "ravdess", "cremad"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", nargs="+", default=list(TAGS.values()), help="svm rf knn mlp")
    ap.add_argument("--params", help="Hafta 3 results.json (sabit hiperparametreler)")
    ap.add_argument("--tune", action="store_true", help="her katta iç doğrulama katıyla hiperparametre seç (yavaş)")
    ap.add_argument("--no_aug", action="store_true", help="artırılmış kopyaları eğitimde de kullanma")
    ap.add_argument("--ablate_extras", action="store_true", help="feat setinin son 4 ek ölçümünü at")
    ap.add_argument("--demographics", help="CREMA-D VideoDemographics.csv (cinsiyet dengeli katlar)")
    ap.add_argument("--include_dir", nargs="*", default=[], help="cnn.py --cv çıktı klasörleri (ortak testlere dahil)")
    ap.add_argument("--out_dir", default="cv_results")
    ap.add_argument("--n_jobs", type=int, default=-1, help="paralel iş sayısı (-1: tüm çekirdekler)")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    pool = pooled_metadata(args.in_dir)
    X = load_pooled(args.feat_dir, args.feat_name, args.in_dir).astype(np.float32)
    if args.feat_name == "feat" and args.ablate_extras:
        X = X[:, :-4]
    y = pool.label.to_numpy()
    folds = make_folds(pool, args.k, args.seed, args.demographics)
    sel = select_dataset(pool, args.dataset)
    is_aug = pool.is_aug.to_numpy()
    check_no_leak(pool, folds, sel)
    sig = fold_signature(pool, folds)
    print(f"Öznitelik: {args.feat_name} {X.shape[1]} boyut | veri: {args.dataset} | k={args.k} | kat imzası {sig}")

    eval_idx = np.where(sel & ~is_aug)[0]              # değerlendirilen orijinal kayıtlar (havuz indeksleri)
    y_oof, f_oof = y[eval_idx], folds[eval_idx]
    ds_oof = pool.dataset.to_numpy()[eval_idx]
    for f in range(args.k):
        spk = pool.speaker[sel & (folds == f)].nunique()
        print(f"  kat {f}: {np.sum(f_oof == f)} test kaydı, {spk} konuşmacı")

    P = fixed_params(args.params)
    names = [n for n, t in TAGS.items() if t in args.models]
    oof = {n: np.full(len(eval_idx), -1) for n in names}
    fold_m = {n: [] for n in names}
    chosen = {n: [] for n in names}
    test_frac = []

    jobs = [(n, f) for f in range(args.k) for n in names]       # her (model, kat) bağımsız bir iş
    n_jobs = args.n_jobs if args.n_jobs > 0 else (os.cpu_count() or 1)
    print(f"{len(jobs)} iş, {min(n_jobs, len(jobs))} paralel çekirdekte çalışıyor...")
    out_jobs = Parallel(n_jobs=min(n_jobs, len(jobs)), verbose=0)(
        delayed(run_job)(n, f, X, y, sel, folds, is_aug, eval_idx, f_oof, P[n], args.tune, args.no_aug, args.k)
        for n, f in jobs)
    for (n, f), (te_pos, pred, m, params, tfrac, secs) in zip(jobs, out_jobs):
        oof[n][te_pos] = pred
        fold_m[n].append(m)
        chosen[n].append({k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()})
        if n == names[0]:
            test_frac.append(tfrac)
        print(f"kat {f} | {n:14s} acc={m['acc']:.3f} F1={m['macro_f1']:.3f} UAR={m['uar']:.3f} "
              f"({secs:.0f} sn) {params if args.tune else ''}")

    # --- CNN vb. dış sonuçların eklenmesi (aynı katlar şart) ---
    for d in args.include_dir:
        d = Path(d)
        s = json.load(open(d / "cv_summary.json"))
        assert s["fold_signature"] == sig, f"{d}: farklı katlar (imza {s['fold_signature']} != {sig})"
        idx = np.load(d / "oof_index.npy")
        assert np.array_equal(idx, eval_idx), f"{d}: değerlendirilen kayıtlar farklı (--dataset aynı mı?)"
        for n, info in s["models"].items():
            oof[n] = np.load(d / info["oof_file"])
            fold_m[n] = info["folds"]
            print("Dahil edildi:", n, "<-", d)
    assert all((p >= 0).all() for p in oof.values())

    # --- Özet ---
    results, summary_rows = {}, []
    for n, pred in oof.items():
        F = fold_m[n]
        agg = {k: (float(np.mean([m[k] for m in F])), float(np.std([m[k] for m in F]))) for k in ["acc", "macro_f1", "uar"]}
        pooled = metrics(y_oof, pred)
        by_ds = {DS_NAMES[ds]: metrics(y_oof[ds_oof == ds], pred[ds_oof == ds]) for ds in np.unique(ds_oof)}
        cm = confusion_matrix(y_oof, pred, labels=LABELS)
        tag = TAGS.get(n, re.sub(r"[^a-z0-9]+", "_", n.lower()).strip("_"))   # ör. "svm_w2v_orta_ort"
        plot_cm(cm, f"{n} - {args.k} katlı CV ({args.feat_name})", out / f"cm_oof_{tag}.png")
        np.save(out / f"oof_{tag}.npy", pred)
        per_class = (cm.diagonal() / np.maximum(cm.sum(1), 1)).tolist()
        results[n] = dict(folds=F, mean_std=agg, pooled_oof=pooled, oof_by_dataset=by_ds,
                          per_class_recall=dict(zip(CLASSES, per_class)), confusion_matrix=cm.tolist(),
                          params=chosen.get(n), oof_file=f"oof_{tag}.npy")

    # --- İstatistiksel testler ---
    mc = pairwise_mcnemar(y_oof, oof)
    tf = float(np.mean(test_frac))
    tt = []
    for a, b in itertools.combinations(oof, 2):
        for metric in ["acc", "macro_f1"]:
            r = paired_t([m[metric] for m in fold_m[a]], [m[metric] for m in fold_m[b]], test_frac=tf)
            r.update(model_1=a, model_2=b, metric=metric)
            tt.append(r)
    for metric in ["acc", "macro_f1"]:
        rows = [r for r in tt if r["metric"] == metric]
        for r, ph in zip(rows, holm([r["p"] for r in rows])):
            r["p_holm"] = ph

    np.save(out / "oof_index.npy", eval_idx); np.save(out / "y_oof.npy", y_oof)
    np.save(out / "dataset_oof.npy", np.array([0 if d == "ravdess" else 1 for d in ds_oof]))
    np.save(out / "fold_oof.npy", f_oof)
    json.dump(dict(feat_name=args.feat_name, n_features=int(X.shape[1]), dataset=args.dataset, k=args.k,
                   seed=args.seed, fold_signature=sig, tuned=args.tune, use_aug=not args.no_aug,
                   n_eval=int(len(eval_idx)), models=results, mcnemar=mc, paired_t=tt),
              open(out / "cv_summary.json", "w"), indent=2, ensure_ascii=False)

    # --- Markdown tablo (rapora doğrudan aktarılabilir) ---
    dsn = sorted({k for r in results.values() for k in r["oof_by_dataset"]})
    L = [f"# {args.k} katlı konuşmacı bazlı CV - öznitelik: {args.feat_name} ({X.shape[1]}), veri: {args.dataset}\n",
         "| Model | Doğruluk (ort ± ss) | Makro F1 (ort ± ss) | UAR (ort ± ss) |" + "".join(f" {d} doğr. |" for d in dsn),
         "|---|---|---|---|" + "---|" * len(dsn)]
    for n, r in results.items():
        a, f1, u = r["mean_std"]["acc"], r["mean_std"]["macro_f1"], r["mean_std"]["uar"]
        L.append(f"| {n} | {a[0]*100:.1f} ± {a[1]*100:.1f}% | {f1[0]:.3f} ± {f1[1]:.3f} | {u[0]:.3f} ± {u[1]:.3f} |"
                 + "".join(f" {r['oof_by_dataset'][d]['acc']*100:.1f}% |" for d in dsn))
    L += ["\n## Sınıf bazında duyarlılık (OOF)\n", "| Model | " + " | ".join(CLASSES) + " |",
          "|---|" + "---|" * len(CLASSES)]
    for n, r in results.items():
        L.append(f"| {n} | " + " | ".join(f"{r['per_class_recall'][c]*100:.0f}%" for c in CLASSES) + " |")
    L += ["\n## McNemar testi (katlar dışı tahminler, Holm düzeltmeli)\n", mcnemar_table(mc),
          "\n## Eşleştirilmiş t-testi (kat doğrulukları)\n",
          "| Model 1 | Model 2 | Ort. fark | p (t-testi) | p (Holm) | p (Nadeau-Bengio) |", "|---|---|---|---|---|---|"]
    for r in [r for r in tt if r["metric"] == "acc"]:
        L.append(f"| {r['model_1']} | {r['model_2']} | {r['mean_diff']*100:+.1f} puan | {fmt_p(r['p'])} | "
                 f"{fmt_p(r['p_holm'])} | {fmt_p(r.get('p_corr', 1.0))} |")
    best = max(results, key=lambda n: results[n]["mean_std"]["acc"][0])
    L.append(f"\nEn iyi model: **{best}** ({results[best]['mean_std']['acc'][0]*100:.1f}%). "
             f"Anlamlılık düzeyi α = {ALPHA}.")
    (out / "cv_table.md").write_text("\n".join(L), encoding="utf-8")
    print("\n" + "\n".join(L))


if __name__ == "__main__":
    main()
