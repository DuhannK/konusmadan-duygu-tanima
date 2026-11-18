# Demo Arayüzünü Çalıştırma (Hafta 6)

Arayüz **Streamlit** ile yazıldı. Google Colab'da **cloudflared tüneli** üzerinden açılır;
tarayıcıdan herkesin erişebileceği geçici bir link verir.

## Gerekenler (eğitilmiş model dosyaları)
Arayüz modelleri sıfırdan eğitmez; önceden kaydedilmiş dosyaları yükler. En az biri gerekli:

| Model | Gereken dosyalar |
|---|---|
| **wav2vec 2.0 + SVM** (en iyi) | `results_w2v/scaler.joblib`, `results_w2v/model_svm.joblib`, `features_v3/extra_features_info.json` |
| CNN (log-mel) | `cnn_all/cnn_best.pt` |
| Klasik (556) | `results_v3/scaler.joblib`, `results_v3/model_svm.joblib` |

Bu dosyalar ekibin Google Drive yedeklerinde (`tertipler_results*`, `tertipler_features*`) var.
Hangi model varsa arayüzde o otomatik görünür; hepsi olması şart değil.

## Colab hücreleri

**1) Depoyu çek ve kütüphaneleri kur**
```python
!git clone https://github.com/DuhannK/konusmadan-duygu-tanima.git
%cd konusmadan-duygu-tanima
!pip -q install -r requirements.txt
!pip -q install streamlit transformers torch cloudflared || pip -q install streamlit transformers torch
```

**2) Model dosyalarını Drive'dan getir** (yollarını kendi yedeğine göre düzelt)
```python
from google.colab import drive
drive.mount('/content/drive')
import shutil, os
BASE = '/content/drive/MyDrive'   # yedeklerin olduğu klasör
# örnek: klasörleri repo köküne kopyala
for src, dst in [
    (f'{BASE}/tertipler_results_w2v', 'results_w2v'),
    (f'{BASE}/tertipler_features_v3', 'features_v3'),
    (f'{BASE}/tertipler_results_v3',  'results_v3'),
    (f'{BASE}/tertipler_cnn_all',     'cnn_all'),
]:
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print('kopyalandı:', dst)
!ls results_w2v features_v3 2>/dev/null
```

**3) cloudflared'i indir** (pip'ten kurulmadıysa)
```python
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared
!chmod +x /usr/local/bin/cloudflared
```

**4) Arayüzü başlat + tünel aç**
```python
import subprocess, time, re, threading
# Streamlit'i arka planda çalıştır
subprocess.Popen(
    "streamlit run src/app.py --server.port 8501 --server.headless true -- "
    "--w2v_dir results_w2v --feat_info features_v3/extra_features_info.json "
    "--cnn cnn_all/cnn_best.pt --classic_dir results_v3",
    shell=True)
time.sleep(8)
# Tünel aç, verdiği linki yazdır
p = subprocess.Popen("cloudflared tunnel --url http://localhost:8501 --no-autoupdate",
                     shell=True, stderr=subprocess.PIPE, text=True)
for line in p.stderr:
    m = re.search(r"https://[-\w.]+trycloudflare\.com", line)
    if m:
        print("\n👉 ARAYÜZ LİNKİ:", m.group(0), "\n")
        break
```

Çıkan `https://...trycloudflare.com` linkine tıkla. Arayüz açılır:
soldan model seç, dosya yükle ya da mikrofonla konuş, tahmini ve olasılıkları gör.

## Notlar
- İlk wav2vec tahmininde `wav2vec2-base` modeli indirilir (~360 MB), birkaç saniye sürer; sonra önbelleğe alınır.
- GPU şart değil; demo CPU'da da çalışır (wav2vec tahmini CPU'da ~1–2 sn).
- Link Colab oturumu kapanınca ölür. Kalıcı link için Streamlit Community Cloud / Hugging Face Spaces'e taşınabilir (ileride).
- Yerel bilgisayarda: `pip install -r requirements.txt streamlit transformers torch` sonra
  `streamlit run src/app.py -- --w2v_dir results_w2v --feat_info features_v3/extra_features_info.json`
