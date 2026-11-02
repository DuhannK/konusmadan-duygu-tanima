# 5 katlı konuşmacı bazlı CV - öznitelik: w2v (768), veri: ravdess

| Model | Doğruluk (ort ± ss) | Makro F1 (ort ± ss) | UAR (ort ± ss) | RAVDESS doğr. |
|---|---|---|---|---|
| SVM (RBF) | 78.3 ± 4.1% | 0.779 ± 0.041 | 0.780 ± 0.039 | 78.2% |
| Random Forest | 67.0 ± 2.5% | 0.636 ± 0.028 | 0.646 ± 0.026 | 67.0% |
| k-NN | 63.8 ± 3.3% | 0.622 ± 0.029 | 0.626 ± 0.032 | 63.7% |
| MLP | 76.3 ± 5.2% | 0.758 ± 0.051 | 0.759 ± 0.052 | 76.2% |

## Sınıf bazında duyarlılık (OOF)

| Model | angry | happy | sad | fear | disgust | neutral |
|---|---|---|---|---|---|---|
| SVM (RBF) | 87% | 69% | 62% | 81% | 84% | 83% |
| Random Forest | 79% | 39% | 26% | 67% | 80% | 97% |
| k-NN | 85% | 54% | 40% | 52% | 66% | 79% |
| MLP | 84% | 68% | 59% | 80% | 80% | 82% |

## McNemar testi (katlar dışı tahminler, Holm düzeltmeli)

| Model 1 | Model 2 | Doğruluk 1 | Doğruluk 2 | b | c | p | p (Holm) | Anlamlı? |
|---|---|---|---|---|---|---|---|---|
| SVM (RBF) | Random Forest | 78.2% | 67.0% | 201 | 61 | < 0,001 | < 0,001 | Evet |
| SVM (RBF) | k-NN | 78.2% | 63.7% | 233 | 52 | < 0,001 | < 0,001 | Evet |
| SVM (RBF) | MLP | 78.2% | 76.2% | 73 | 48 | 0,029 | 0,033 | Evet |
| Random Forest | k-NN | 67.0% | 63.7% | 160 | 119 | 0,017 | 0,033 | Evet |
| Random Forest | MLP | 67.0% | 76.2% | 76 | 191 | < 0,001 | < 0,001 | Evet |
| k-NN | MLP | 63.7% | 76.2% | 75 | 231 | < 0,001 | < 0,001 | Evet |

## Eşleştirilmiş t-testi (kat doğrulukları)

| Model 1 | Model 2 | Ort. fark | p (t-testi) | p (Holm) | p (Nadeau-Bengio) |
|---|---|---|---|---|---|
| SVM (RBF) | Random Forest | +11.3 puan | < 0,001 | 0,002 | 0,001 |
| SVM (RBF) | k-NN | +14.5 puan | < 0,001 | < 0,001 | < 0,001 |
| SVM (RBF) | MLP | +2.0 puan | 0,220 | 0,220 | 0,335 |
| Random Forest | k-NN | +3.2 puan | 0,030 | 0,059 | 0,067 |
| Random Forest | MLP | -9.3 puan | 0,006 | 0,019 | 0,017 |
| k-NN | MLP | -12.5 puan | 0,005 | 0,019 | 0,013 |

En iyi model: **SVM (RBF)** (78.3%). Anlamlılık düzeyi α = 0.05.