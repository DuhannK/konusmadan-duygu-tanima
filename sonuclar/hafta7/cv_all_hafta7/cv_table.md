# 5 katlı konuşmacı bazlı CV - öznitelik: feat (556), veri: all

| Model | Doğruluk (ort ± ss) | Makro F1 (ort ± ss) | UAR (ort ± ss) | CREMA-D doğr. | RAVDESS doğr. |
|---|---|---|---|---|---|
| SVM (RBF) | 52.7 ± 1.4% | 0.523 ± 0.017 | 0.527 ± 0.014 | 53.1% | 49.8% |
| CNN | 65.8 ± 1.8% | 0.656 ± 0.018 | 0.659 ± 0.018 | 65.2% | 69.3% |
| CNN-LSTM | 66.0 ± 1.0% | 0.660 ± 0.010 | 0.661 ± 0.010 | 65.7% | 67.7% |
| SVM-w2v (tek_en_iyi) | 70.5 ± 1.8% | 0.704 ± 0.018 | 0.706 ± 0.018 | 69.4% | 77.1% |
| SVM-w2v (tum_ort) | 72.0 ± 2.6% | 0.720 ± 0.026 | 0.720 ± 0.026 | 71.1% | 77.3% |
| SVM-w2v (orta_ort) | 71.2 ± 2.2% | 0.711 ± 0.022 | 0.713 ± 0.022 | 70.0% | 78.2% |
| SVM-w2v (en_iyi_3) | 71.3 ± 2.0% | 0.712 ± 0.020 | 0.714 ± 0.021 | 70.2% | 78.0% |

## Sınıf bazında duyarlılık (OOF)

| Model | angry | happy | sad | fear | disgust | neutral |
|---|---|---|---|---|---|---|
| SVM (RBF) | 69% | 51% | 56% | 43% | 42% | 55% |
| CNN | 76% | 64% | 58% | 61% | 59% | 77% |
| CNN-LSTM | 77% | 61% | 59% | 62% | 63% | 74% |
| SVM-w2v (tek_en_iyi) | 83% | 68% | 65% | 60% | 65% | 82% |
| SVM-w2v (tum_ort) | 83% | 72% | 67% | 64% | 67% | 80% |
| SVM-w2v (orta_ort) | 83% | 68% | 68% | 61% | 65% | 83% |
| SVM-w2v (en_iyi_3) | 84% | 68% | 67% | 61% | 65% | 83% |

## McNemar testi (katlar dışı tahminler, Holm düzeltmeli)

| Model 1 | Model 2 | Doğruluk 1 | Doğruluk 2 | b | c | p | p (Holm) | Anlamlı? |
|---|---|---|---|---|---|---|---|---|
| SVM (RBF) | CNN | 52.6% | 65.8% | 770 | 1911 | < 0,001 | < 0,001 | Evet |
| SVM (RBF) | CNN-LSTM | 52.6% | 66.0% | 780 | 1941 | < 0,001 | < 0,001 | Evet |
| SVM (RBF) | SVM-w2v (tek_en_iyi) | 52.6% | 70.5% | 561 | 2114 | < 0,001 | < 0,001 | Evet |
| SVM (RBF) | SVM-w2v (tum_ort) | 52.6% | 72.0% | 581 | 2261 | < 0,001 | < 0,001 | Evet |
| SVM (RBF) | SVM-w2v (orta_ort) | 52.6% | 71.2% | 538 | 2152 | < 0,001 | < 0,001 | Evet |
| SVM (RBF) | SVM-w2v (en_iyi_3) | 52.6% | 71.3% | 555 | 2177 | < 0,001 | < 0,001 | Evet |
| CNN | CNN-LSTM | 65.8% | 66.0% | 681 | 701 | 0,609 | 1,000 | Hayır |
| CNN | SVM-w2v (tek_en_iyi) | 65.8% | 70.5% | 785 | 1197 | < 0,001 | < 0,001 | Evet |
| CNN | SVM-w2v (tum_ort) | 65.8% | 72.0% | 754 | 1293 | < 0,001 | < 0,001 | Evet |
| CNN | SVM-w2v (orta_ort) | 65.8% | 71.2% | 733 | 1206 | < 0,001 | < 0,001 | Evet |
| CNN | SVM-w2v (en_iyi_3) | 65.8% | 71.3% | 742 | 1223 | < 0,001 | < 0,001 | Evet |
| CNN-LSTM | SVM-w2v (tek_en_iyi) | 66.0% | 70.5% | 819 | 1211 | < 0,001 | < 0,001 | Evet |
| CNN-LSTM | SVM-w2v (tum_ort) | 66.0% | 72.0% | 780 | 1299 | < 0,001 | < 0,001 | Evet |
| CNN-LSTM | SVM-w2v (orta_ort) | 66.0% | 71.2% | 763 | 1216 | < 0,001 | < 0,001 | Evet |
| CNN-LSTM | SVM-w2v (en_iyi_3) | 66.0% | 71.3% | 779 | 1240 | < 0,001 | < 0,001 | Evet |
| SVM-w2v (tek_en_iyi) | SVM-w2v (tum_ort) | 70.5% | 72.0% | 522 | 649 | < 0,001 | 0,002 | Evet |
| SVM-w2v (tek_en_iyi) | SVM-w2v (orta_ort) | 70.5% | 71.2% | 363 | 424 | 0,032 | 0,141 | Hayır |
| SVM-w2v (tek_en_iyi) | SVM-w2v (en_iyi_3) | 70.5% | 71.3% | 345 | 414 | 0,014 | 0,081 | Hayır |
| SVM-w2v (tum_ort) | SVM-w2v (orta_ort) | 72.0% | 71.2% | 472 | 406 | 0,028 | 0,141 | Hayır |
| SVM-w2v (tum_ort) | SVM-w2v (en_iyi_3) | 72.0% | 71.3% | 564 | 506 | 0,081 | 0,244 | Hayır |
| SVM-w2v (orta_ort) | SVM-w2v (en_iyi_3) | 71.2% | 71.3% | 246 | 254 | 0,754 | 1,000 | Hayır |

## Eşleştirilmiş t-testi (kat doğrulukları)

| Model 1 | Model 2 | Ort. fark | p (t-testi) | p (Holm) | p (Nadeau-Bengio) |
|---|---|---|---|---|---|
| SVM (RBF) | CNN | -13.1 puan | < 0,001 | < 0,001 | < 0,001 |
| SVM (RBF) | CNN-LSTM | -13.3 puan | < 0,001 | < 0,001 | < 0,001 |
| SVM (RBF) | SVM-w2v (tek_en_iyi) | -17.8 puan | < 0,001 | < 0,001 | < 0,001 |
| SVM (RBF) | SVM-w2v (tum_ort) | -19.3 puan | < 0,001 | < 0,001 | < 0,001 |
| SVM (RBF) | SVM-w2v (orta_ort) | -18.6 puan | < 0,001 | < 0,001 | < 0,001 |
| SVM (RBF) | SVM-w2v (en_iyi_3) | -18.6 puan | < 0,001 | < 0,001 | < 0,001 |
| CNN | CNN-LSTM | -0.2 puan | 0,740 | 1,000 | 0,820 |
| CNN | SVM-w2v (tek_en_iyi) | -4.7 puan | < 0,001 | 0,002 | < 0,001 |
| CNN | SVM-w2v (tum_ort) | -6.2 puan | < 0,001 | 0,004 | 0,001 |
| CNN | SVM-w2v (orta_ort) | -5.4 puan | < 0,001 | < 0,001 | < 0,001 |
| CNN | SVM-w2v (en_iyi_3) | -5.5 puan | < 0,001 | < 0,001 | < 0,001 |
| CNN-LSTM | SVM-w2v (tek_en_iyi) | -4.5 puan | 0,002 | 0,017 | 0,007 |
| CNN-LSTM | SVM-w2v (tum_ort) | -6.0 puan | 0,003 | 0,027 | 0,012 |
| CNN-LSTM | SVM-w2v (orta_ort) | -5.2 puan | 0,004 | 0,029 | 0,014 |
| CNN-LSTM | SVM-w2v (en_iyi_3) | -5.3 puan | 0,002 | 0,017 | 0,006 |
| SVM-w2v (tek_en_iyi) | SVM-w2v (tum_ort) | -1.5 puan | 0,064 | 0,445 | 0,157 |
| SVM-w2v (tek_en_iyi) | SVM-w2v (orta_ort) | -0.7 puan | 0,270 | 1,000 | 0,432 |
| SVM-w2v (tek_en_iyi) | SVM-w2v (en_iyi_3) | -0.8 puan | 0,077 | 0,464 | 0,182 |
| SVM-w2v (tum_ort) | SVM-w2v (orta_ort) | +0.8 puan | 0,258 | 1,000 | 0,419 |
| SVM-w2v (tum_ort) | SVM-w2v (en_iyi_3) | +0.7 puan | 0,163 | 0,814 | 0,308 |
| SVM-w2v (orta_ort) | SVM-w2v (en_iyi_3) | -0.1 puan | 0,775 | 1,000 | 0,845 |

En iyi model: **SVM-w2v (tum_ort)** (72.0%). Anlamlılık düzeyi α = 0.05.