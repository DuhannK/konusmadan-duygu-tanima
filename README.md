# Konuşma Kayıtlarından Duygu Sınıflandırılması (SER)

Fırat Üniversitesi – Bilgisayar Mühendisliği Tasarım Dersi projesi.
**Ekip: Tertipler** — Duhan HARBİ, Cihat DURMAZ, Samet Uğur ŞANLIDAĞ

Kısa (2–5 sn) tek konuşmacılı ses kayıtlarını altı duygu sınıfından birine atar:
**öfke, mutluluk, üzüntü, korku, iğrenme, nötr.** Yöntem akustik özniteliklere
dayanır, kelimelerin anlamına değil; bu yüzden büyük ölçüde dilden bağımsızdır.

## Veri Setleri
- **RAVDESS** [Zenodo/Kaggle] — 24 oyuncu, 1440 kayıt (sakin→nötr, şaşkınlık çıkarıldı → 6 sınıf).
- **CREMA-D** — 91 oyuncu, 7442 kayıt, 6 sınıf.

Ham veri depoya **yüklenmez** (bkz. `.gitignore`); betikler Kaggle'dan indirip işler.

## Akış
```
Ham ses → preprocess.py → features.py / extra_features.py → baseline.py, cnn.py
        → cv_eval.py (5 katlı CV + istatistiksel testler) → app.py (Streamlit demo)
```

## Betikler (`src/`)
| Betik | Görevi |
|---|---|
| `preprocess.py` | 16 kHz mono, sessizlik kesme, normalizasyon, 3 sn sabitleme, 6 sınıfa etiketleme, veri artırma, konuşmacıdan bağımsız bölümleme. RAVDESS + CREMA-D destekler. |
| `features.py` | Her kayıttan 556 öznitelik (MFCC+Δ+ΔΔ, ZCR, RMS, centroid, bandwidth, roll-off, chroma, F0 + 4 ek ölçüm) ve CNN için log-mel spektrogram. |
| `baseline.py` | SVM, Random Forest, k-NN, MLP; makro F1 / UAR / karışıklık matrisi; veri seti bazlı raporlama. `--feat_name` ile farklı öznitelik setleri; test tahminlerini kaydeder. |
| `folds.py` | *(Hafta 4)* Konuşmacı bazlı, cinsiyet dengeli k-katlı bölümleme (tüm betiklerde aynı katlar). |
| `cv_eval.py` | *(Hafta 4, H5)* Konuşmacı bazlı 5 katlı CV (ort. ± ss), McNemar ve eşleştirilmiş t-testi (Holm, Nadeau-Bengio). |
| `stats_tests.py` | *(Hafta 4, H5)* McNemar / t-testi fonksiyonları; tek test kümesinde model tahminlerini karşılaştırma. |
| `extra_features.py` | *(Hafta 4)* eGeMAPS (88), ComParE (6373) — openSMILE; wav2vec 2.0 gömmeleri (katman val ile seçilir). |
| `cnn.py` | *(Hafta 4, H4)* Log-mel (128×301) girdili 2B CNN; SpecAugment, erken durdurma; `--cv 5` ile CV. |
| `app.py` | *(Hafta 4, H6)* Streamlit demo: dosya yükleme + mikrofon, sınıf olasılıkları. |

## Kullanım
```bash
pip install -r requirements.txt

# 1) Ön işleme (RAVDESS + CREMA-D)
python src/preprocess.py \
  --ravdess_dir <RAVDESS_yolu> \
  --cremad_dir <CREMA-D_AudioWAV_yolu> \
  --demographics VideoDemographics.csv \
  --out_dir processed_v3 --augment

# 2) Öznitelik çıkarımı
python src/features.py --in_dir processed_v3 --out_dir features_v3 --skip_logmel

# 3) Klasik modeller
python src/baseline.py --feat_dir features_v3 --out_dir results_v3
```

### Hafta 4
```bash
pip install -r requirements.txt

# 4) Standart öznitelik setleri + wav2vec 2.0 (GPU önerilir)
python src/extra_features.py --in_dir processed_v3 --out_dir features_v3 --sets egemaps compare w2v

# 5) Tek bölümlemede (train/val/test) farklı öznitelik setleri
python src/baseline.py --feat_dir features_v3 --feat_name egemaps --out_dir results_egemaps
python src/baseline.py --feat_dir features_v3 --feat_name w2v     --out_dir results_w2v

# 6) CNN (H4) - yalnızca RAVDESS ve birleşik
python src/cnn.py --in_dir processed_v3 --dataset ravdess --out_dir cnn_ravdess
python src/cnn.py --in_dir processed_v3 --dataset all     --out_dir cnn_all

# 7) Konuşmacı bazlı 5 katlı CV + istatistiksel testler (H5)
python src/cnn.py --in_dir processed_v3 --dataset ravdess --cv 5 --epochs 40 --out_dir cnn_cv_ravdess
python src/cv_eval.py --in_dir processed_v3 --feat_dir features_v3 --dataset ravdess \
    --params results_v3/results.json --include_dir cnn_cv_ravdess --out_dir cv_ravdess_feat
python src/cv_eval.py --in_dir processed_v3 --feat_dir features_v3 --feat_name w2v --dataset ravdess --out_dir cv_ravdess_w2v

# 8) Tek test kümesinde CNN - SVM McNemar
python src/stats_tests.py --y cnn_all/y_test.npy --pred SVM=results_v3/pred_svm.npy CNN=cnn_all/pred_test.npy \
    --mask_dataset cnn_all/dataset_test.npy

# 9) Demo (H6)
streamlit run src/app.py -- --cnn cnn_all/cnn_best.pt --classic_dir results_v3
```

## Güncel Sonuçlar (Hafta 3)
Konuşmacıdan bağımsız test:
- **Yalnızca RAVDESS:** en iyi %53,8 (Random Forest) / %53,4 (SVM).
- **RAVDESS + CREMA-D birleşik:** SVM genel %49,4 (CREMA-D %50,2, RAVDESS %44,7).

Hedef (H3 ≥ %55) henüz tutmadı; sonraki adımlar: çapraz doğrulama + istatistiksel test (H5),
CNN (H4), demo (H6).

## Hedefler
H1 Veri hazırlama · H2 Öznitelik · H3 Klasik model (≥%55) · H4 Derin model (CNN/CNN-LSTM, ≥%65)
· H5 Karşılaştırmalı değerlendirme · H6 Streamlit/Gradio demo.
