# Hafta 4 Sonuçları

Colab (T4 GPU) çalıştırmalarının çıktıları. Konsol çıktılarının tamamı `logs/` klasöründedir.
Büyük dosyalar (tahmin dizileri `.npy`, modeller `.pt`/`.joblib`, veri) depoya eklenmemiştir.

| Klasör / dosya | İçerik |
|---|---|
| `klasik_v3/` | SVM, RF, k-NN, MLP - 556 el yapımı öznitelik, tek bölümleme (test) |
| `klasik_egemaps/` | Aynı modeller - eGeMAPS (88 öznitelik) |
| `klasik_w2v/` | Aynı modeller - wav2vec 2.0 gömmeleri (768 boyut, katman 5) |
| `cnn_ravdess/`, `cnn_all/` | CNN - tek bölümleme (yalnızca RAVDESS / birleşik); eğitim eğrisi ve karışıklık matrisi |
| `cnn_cv_ravdess/` | CNN - RAVDESS konuşmacı bazlı 5 katlı CV |
| `cv_ravdess_feat/` | 5 katlı CV, RAVDESS, 556 öznitelik + CNN; McNemar ve t-testleri (`cv_table.md`) |
| `cv_all_feat/` | 5 katlı CV, birleşik veri, 556 öznitelik (`cv_table.md`) |
| `cv_ravdess_w2v/` | 5 katlı CV, RAVDESS, wav2vec 2.0 (`cv_table.md`) |
| `ek_mcnemar_test.md` / `.json` | Birleşik test kümesinde öznitelik setleri ve CNN arasında McNemar testleri |

Özet: en iyi sonuç wav2vec 2.0 + SVM (birleşik test %69,7, makro F1 0,696; RAVDESS CV %78,3 ± 4,1).
Ayrıntılar Hafta 4 raporunun 4.5 bölümündedir.
