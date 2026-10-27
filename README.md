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
Ham ses → preprocess.py → features.py → baseline.py → (değerlendirme, demo)
```

## Betikler (`src/`)
| Betik | Görevi |
|---|---|
| `preprocess.py` | 16 kHz mono, sessizlik kesme, normalizasyon, 3 sn sabitleme, 6 sınıfa etiketleme, veri artırma, konuşmacıdan bağımsız bölümleme. RAVDESS + CREMA-D destekler. |
| `features.py` | Her kayıttan 556 öznitelik (MFCC+Δ+ΔΔ, ZCR, RMS, centroid, bandwidth, roll-off, chroma, F0 + 4 ek ölçüm) ve CNN için log-mel spektrogram. |
| `baseline.py` | SVM, Random Forest, k-NN, MLP; makro F1 / UAR / karışıklık matrisi; veri seti bazlı raporlama. |

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

## Güncel Sonuçlar (Hafta 3)
Konuşmacıdan bağımsız test:
- **Yalnızca RAVDESS:** en iyi %53,8 (Random Forest) / %53,4 (SVM).
- **RAVDESS + CREMA-D birleşik:** SVM genel %49,4 (CREMA-D %50,2, RAVDESS %44,7).

Hedef (H3 ≥ %55) henüz tutmadı; sonraki adımlar: çapraz doğrulama + istatistiksel test (H5),
CNN (H4), demo (H6).

## Hedefler
H1 Veri hazırlama · H2 Öznitelik · H3 Klasik model (≥%55) · H4 Derin model (CNN/CNN-LSTM, ≥%65)
· H5 Karşılaştırmalı değerlendirme · H6 Streamlit/Gradio demo.
