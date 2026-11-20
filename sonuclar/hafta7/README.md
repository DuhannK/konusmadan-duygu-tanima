# Hafta 7 Sonuçları

Colab (L4 GPU) çalıştırmalarının çıktıları, `colab/hafta7_colab.ipynb` ile üretildi. Konsol çıktıları `logs/` klasöründedir.
Büyük dosyalar (tahmin dizileri `.npy`, modeller `.pt`/`.joblib`, veri) depoya eklenmemiştir.

| Klasör | İçerik |
|---|---|
| `cnn_all/`, `cnn_lstm_all/` | CNN / CNN-LSTM, birleşik veri, tek bölümleme: sonuçlar, eğitim eğrisi, karışıklık matrisi |
| `cnn_cv_all/`, `cnn_lstm_cv_all/` | CNN / CNN-LSTM, birleşik veri, konuşmacı bazlı 5 katlı CV |
| `w2v_exp_all/` | wav2vec 2.0 katman birleşimi deneyleri (`w2v_table.md`) |
| `cv_all_hafta7/` | Tüm modeller aynı katlarda: sınıf duyarlılıkları, McNemar ve t-testleri (`cv_table.md`) |
| `logs/` | Tüm adımların konsol çıktıları ve kat günlükleri (`kat*_*.log`) |

## Özet (birleşik veri, 5 katlı CV)

| Model | Doğruluk | Makro F1 |
|---|---|---|
| SVM (556 el yapımı öznitelik) | %52,7 ± 1,4 | 0,523 |
| CNN | %65,8 ± 1,8 | 0,656 |
| CNN-LSTM | %66,0 ± 1,0 | 0,660 |
| SVM + wav2vec 2.0 (tüm katmanların ortalaması) | %72,0 ± 2,6 | 0,720 |

CNN ile CNN-LSTM arasındaki fark anlamlı değildir; wav2vec stratejileri iki derin modelden de anlamlı olarak daha iyidir.
Ayrıntılar: `raporlar/Hafta7_Proje_Raporu_Tertipler.docx`, bölüm 4.8.
