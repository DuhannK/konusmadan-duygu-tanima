# 5 katlı konuşmacı bazlı CV - öznitelik: feat (556), veri: ravdess

| Model | Doğruluk (ort ± ss) | Makro F1 (ort ± ss) | UAR (ort ± ss) | RAVDESS doğr. |
|---|---|---|---|---|
| SVM (RBF) | 50.9 ± 5.0% | 0.495 ± 0.048 | 0.501 ± 0.046 | 51.1% |
| Random Forest | 51.1 ± 3.8% | 0.451 ± 0.037 | 0.481 ± 0.041 | 51.0% |
| k-NN | 41.6 ± 2.5% | 0.379 ± 0.020 | 0.391 ± 0.022 | 41.6% |
| MLP | 52.7 ± 6.1% | 0.508 ± 0.062 | 0.515 ± 0.061 | 52.7% |
| CNN | 69.8 ± 5.6% | 0.691 ± 0.055 | 0.692 ± 0.054 | 69.7% |

## Sınıf bazında duyarlılık (OOF)

| Model | angry | happy | sad | fear | disgust | neutral |
|---|---|---|---|---|---|---|
| SVM (RBF) | 58% | 47% | 34% | 50% | 49% | 62% |
| Random Forest | 69% | 38% | 6% | 37% | 50% | 88% |
| k-NN | 49% | 27% | 15% | 40% | 32% | 72% |
| MLP | 65% | 44% | 30% | 52% | 52% | 67% |
| CNN | 73% | 62% | 54% | 73% | 75% | 77% |

## McNemar testi (katlar dışı tahminler, Holm düzeltmeli)

| Model 1 | Model 2 | Doğruluk 1 | Doğruluk 2 | b | c | p | p (Holm) | Anlamlı? |
|---|---|---|---|---|---|---|---|---|
| SVM (RBF) | Random Forest | 51.1% | 51.0% | 182 | 181 | 1,000 | 1,000 | Hayır |
| SVM (RBF) | k-NN | 51.1% | 41.6% | 267 | 148 | < 0,001 | < 0,001 | Evet |
| SVM (RBF) | MLP | 51.1% | 52.7% | 103 | 123 | 0,206 | 0,619 | Hayır |
| SVM (RBF) | CNN | 51.1% | 69.7% | 104 | 336 | < 0,001 | < 0,001 | Evet |
| Random Forest | k-NN | 51.0% | 41.6% | 227 | 109 | < 0,001 | < 0,001 | Evet |
| Random Forest | MLP | 51.0% | 52.7% | 160 | 181 | 0,279 | 0,619 | Hayır |
| Random Forest | CNN | 51.0% | 69.7% | 103 | 336 | < 0,001 | < 0,001 | Evet |
| k-NN | MLP | 41.6% | 52.7% | 124 | 263 | < 0,001 | < 0,001 | Evet |
| k-NN | CNN | 41.6% | 69.7% | 86 | 437 | < 0,001 | < 0,001 | Evet |
| MLP | CNN | 52.7% | 69.7% | 96 | 308 | < 0,001 | < 0,001 | Evet |

## Eşleştirilmiş t-testi (kat doğrulukları)

| Model 1 | Model 2 | Ort. fark | p (t-testi) | p (Holm) | p (Nadeau-Bengio) |
|---|---|---|---|---|---|
| SVM (RBF) | Random Forest | -0.2 puan | 0,951 | 1,000 | 0,963 |
| SVM (RBF) | k-NN | +9.3 puan | 0,014 | 0,084 | 0,034 |
| SVM (RBF) | MLP | -1.7 puan | 0,316 | 0,948 | 0,436 |
| SVM (RBF) | CNN | -18.8 puan | 0,002 | 0,011 | 0,004 |
| Random Forest | k-NN | +9.5 puan | 0,023 | 0,113 | 0,053 |
| Random Forest | MLP | -1.6 puan | 0,519 | 1,000 | 0,622 |
| Random Forest | CNN | -18.7 puan | < 0,001 | 0,001 | < 0,001 |
| k-NN | MLP | -11.1 puan | 0,023 | 0,113 | 0,053 |
| k-NN | CNN | -28.2 puan | 0,001 | 0,008 | 0,003 |
| MLP | CNN | -17.1 puan | < 0,001 | 0,002 | < 0,001 |

En iyi model: **CNN** (69.8%). Anlamlılık düzeyi α = 0.05.