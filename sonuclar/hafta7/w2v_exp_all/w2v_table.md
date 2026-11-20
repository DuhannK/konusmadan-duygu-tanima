# wav2vec 2.0 katman birleşimi - veri: all

| Strateji | Katmanlar | Val makro F1 | Test doğruluk | Test makro F1 | CV doğruluk |
|---|---|---|---|---|---|
| tek_en_iyi | [5] | 0.710 | 68.2% | 0.681 | 70.5 ± 1.8% |
| tum_ort | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] | 0.727 | 69.8% | 0.696 | 72.0 ± 2.6% |
| orta_ort ✓ | [3, 4, 5, 6, 7, 8, 9] | 0.731 | 68.6% | 0.683 | 71.2 ± 2.2% |
| en_iyi_3 | [5, 6, 7] | 0.723 | 69.4% | 0.692 | 71.3 ± 2.0% |

✓ = doğrulama makro F1'ine göre seçilen strateji. Arayüz modeli: `w2v_exp_all/model`