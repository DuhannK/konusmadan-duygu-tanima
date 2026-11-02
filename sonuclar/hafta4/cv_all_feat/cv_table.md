# 5 katlı konuşmacı bazlı CV - öznitelik: feat (556), veri: all

| Model | Doğruluk (ort ± ss) | Makro F1 (ort ± ss) | UAR (ort ± ss) | CREMA-D doğr. | RAVDESS doğr. |
|---|---|---|---|---|---|
| SVM (RBF) | 52.7 ± 1.4% | 0.523 ± 0.017 | 0.527 ± 0.014 | 53.1% | 49.8% |
| Random Forest | 50.0 ± 1.6% | 0.485 ± 0.021 | 0.501 ± 0.016 | 50.2% | 48.5% |
| k-NN | 42.3 ± 0.8% | 0.413 ± 0.008 | 0.424 ± 0.009 | 42.7% | 39.9% |
| MLP | 51.6 ± 1.5% | 0.515 ± 0.015 | 0.516 ± 0.015 | 52.0% | 49.3% |

## Sınıf bazında duyarlılık (OOF)

| Model | angry | happy | sad | fear | disgust | neutral |
|---|---|---|---|---|---|---|
| SVM (RBF) | 69% | 51% | 56% | 43% | 42% | 55% |
| Random Forest | 73% | 45% | 57% | 24% | 37% | 64% |
| k-NN | 60% | 33% | 58% | 27% | 26% | 52% |
| MLP | 65% | 51% | 53% | 43% | 44% | 53% |

## McNemar testi (katlar dışı tahminler, Holm düzeltmeli)

| Model 1 | Model 2 | Doğruluk 1 | Doğruluk 2 | b | c | p | p (Holm) | Anlamlı? |
|---|---|---|---|---|---|---|---|---|
| SVM (RBF) | Random Forest | 52.6% | 50.0% | 1126 | 893 | < 0,001 | < 0,001 | Evet |
| SVM (RBF) | k-NN | 52.6% | 42.3% | 1825 | 926 | < 0,001 | < 0,001 | Evet |
| SVM (RBF) | MLP | 52.6% | 51.6% | 809 | 718 | 0,021 | 0,021 | Evet |
| Random Forest | k-NN | 50.0% | 42.3% | 1507 | 841 | < 0,001 | < 0,001 | Evet |
| Random Forest | MLP | 50.0% | 51.6% | 1127 | 1269 | 0,004 | 0,008 | Evet |
| k-NN | MLP | 42.3% | 51.6% | 1062 | 1870 | < 0,001 | < 0,001 | Evet |

## Eşleştirilmiş t-testi (kat doğrulukları)

| Model 1 | Model 2 | Ort. fark | p (t-testi) | p (Holm) | p (Nadeau-Bengio) |
|---|---|---|---|---|---|
| SVM (RBF) | Random Forest | +2.7 puan | < 0,001 | 0,002 | 0,002 |
| SVM (RBF) | k-NN | +10.4 puan | < 0,001 | 0,002 | 0,001 |
| SVM (RBF) | MLP | +1.1 puan | 0,179 | 0,179 | 0,328 |
| Random Forest | k-NN | +7.7 puan | 0,001 | 0,004 | 0,006 |
| Random Forest | MLP | -1.6 puan | 0,085 | 0,170 | 0,194 |
| k-NN | MLP | -9.3 puan | < 0,001 | < 0,001 | < 0,001 |

En iyi model: **SVM (RBF)** (52.7%). Anlamlılık düzeyi α = 0.05.