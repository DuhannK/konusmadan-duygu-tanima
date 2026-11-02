"""
Konuşmadan Duygu Tanıma - İstatistiksel anlamlılık testleri  |  Takım: Tertipler   (rapor Bölüm 2.7)

  McNemar testi     : İki model AYNI örnekler üzerinde test edildiğinde, yalnızca birinin doğru bildiği
                      örnek sayılarını (b, c) karşılaştırır. b + c < 25 ise kesin (binom) test, aksi hâlde
                      süreklilik düzeltmeli ki-kare kullanılır.
  Eşleştirilmiş t   : k kattaki doğruluk farklarının ortalaması sıfırdan farklı mı? (scipy ttest_rel)
  Düzeltilmiş t     : Nadeau & Bengio (2003). Katların eğitim kümeleri örtüştüğü için klasik t-testi
                      fazla "iyimser"dir; varyans (1/k + n_test/n_train) ile büyütülür. Daha temkinli sonuç verir.
  Holm düzeltmesi   : Çok sayıda model çifti test edildiğinde yanlış pozitif riskini kontrol eder.

Tek bölümleme (train/val/test) üzerindeki tahminleri karşılaştırmak için komut satırı:
  python src/stats_tests.py --y results_v3/y_test.npy \
      --pred SVM=results_v3/pred_svm.npy CNN=cnn_results/pred_test.npy [--mask_dataset results_v3/dataset_test.npy]
"""
import argparse, itertools, json
from pathlib import Path
import numpy as np
from scipy import stats

ALPHA = 0.05


def mcnemar(y, p1, p2):
    c1, c2 = (p1 == y), (p2 == y)
    b = int(np.sum(c1 & ~c2))     # yalnızca model 1 doğru
    c = int(np.sum(~c1 & c2))     # yalnızca model 2 doğru
    n = b + c
    if n == 0:
        return dict(b=b, c=c, stat=0.0, p=1.0, method="yok (fark yok)")
    if n < 25:
        p = stats.binomtest(b, n, 0.5).pvalue
        return dict(b=b, c=c, stat=float(min(b, c)), p=float(p), method="kesin binom")
    chi2 = (abs(b - c) - 1) ** 2 / n
    return dict(b=b, c=c, stat=float(chi2), p=float(stats.chi2.sf(chi2, 1)), method="ki-kare (süreklilik düz.)")


def paired_t(a, b, test_frac=None):
    """a, b: kat bazında skorlar. test_frac = n_test / n_train (düzeltilmiş t için)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    k = len(d)
    out = dict(mean_diff=float(d.mean()), k=k)
    if k < 2 or np.allclose(d, d[0]):
        out.update(t=0.0, p=1.0, t_corr=0.0, p_corr=1.0)
        return out
    t, p = stats.ttest_rel(a, b)
    out.update(t=float(t), p=float(p))
    if test_frac is not None:
        var = d.var(ddof=1)
        t_c = d.mean() / np.sqrt((1 / k + test_frac) * var)
        out.update(t_corr=float(t_c), p_corr=float(2 * stats.t.sf(abs(t_c), k - 1)))
    return out


def holm(pvals):
    """Holm-Bonferroni düzeltilmiş p-değerleri (girdi sırasıyla)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def pairwise_mcnemar(y, preds):
    """preds: {ad: tahmin}. Tüm çiftler için McNemar + Holm."""
    rows = []
    for a, b in itertools.combinations(preds, 2):
        r = mcnemar(y, preds[a], preds[b])
        r.update(model_1=a, model_2=b, acc_1=float(np.mean(preds[a] == y)), acc_2=float(np.mean(preds[b] == y)))
        rows.append(r)
    for r, ph in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = ph
        r["significant"] = bool(ph < ALPHA)
    return rows


def fmt_p(p):
    return "< 0,001" if p < 1e-3 else f"{p:.3f}".replace(".", ",")


def mcnemar_table(rows):
    lines = ["| Model 1 | Model 2 | Doğruluk 1 | Doğruluk 2 | b | c | p | p (Holm) | Anlamlı? |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model_1']} | {r['model_2']} | {r['acc_1']*100:.1f}% | {r['acc_2']*100:.1f}% | "
                     f"{r['b']} | {r['c']} | {fmt_p(r['p'])} | {fmt_p(r['p_holm'])} | "
                     f"{'Evet' if r['significant'] else 'Hayır'} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--y", required=True, help="gerçek etiketler (.npy)")
    ap.add_argument("--pred", nargs="+", required=True, help="AD=yol.npy biçiminde tahmin dosyaları")
    ap.add_argument("--mask_dataset", help="dataset_test.npy: sonuçları veri seti bazında da ver")
    ap.add_argument("--out", help="JSON çıktısı (isteğe bağlı)")
    args = ap.parse_args()

    y = np.load(args.y)
    preds = {}
    for item in args.pred:
        name, path = item.split("=", 1)
        preds[name] = np.load(path)
        assert len(preds[name]) == len(y), f"{name}: tahmin sayısı ({len(preds[name])}) etiket sayısıyla ({len(y)}) uyuşmuyor"

    report = {"tümü": pairwise_mcnemar(y, preds)}
    print("## McNemar testi - tüm test kümesi\n" + mcnemar_table(report["tümü"]))
    if args.mask_dataset:
        ds = np.load(args.mask_dataset)
        for k, nm in {0: "RAVDESS", 1: "CREMA-D"}.items():
            m = ds == k
            if m.any():
                report[nm] = pairwise_mcnemar(y[m], {n: p[m] for n, p in preds.items()})
                print(f"\n## McNemar testi - {nm}\n" + mcnemar_table(report[nm]))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(report, open(args.out, "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
