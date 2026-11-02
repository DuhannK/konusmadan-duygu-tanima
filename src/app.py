"""
Konuşmadan Duygu Tanıma - Demo uygulaması (H6)  |  Takım: Tertipler

Dosya yükleyerek ya da mikrofonla konuşarak duygu tahmini yapar ve her sınıfın olasılığını gösterir.
Ses, eğitimdeki ile AYNI ön işlemeden geçer (preprocess.load_and_clean: 16 kHz mono, sessizlik kesme,
genlik normalizasyonu, 3 sn). 3,5 sn'den uzun kayıtlar 3 sn'lik örtüşen pencerelere bölünür ve
pencerelerin olasılıkları ortalanır.

Modeller:
  CNN     : cnn.py çıktısı  (ör. cnn_all/cnn_best.pt)
  Klasik  : baseline.py çıktısı (scaler.joblib + model_*.joblib, 556 boyutlu Hafta 3 öznitelikleri)

Çalıştırma:
  streamlit run src/app.py -- --cnn cnn_all/cnn_best.pt --classic_dir results_v3
"""
import argparse, sys, tempfile, time
from pathlib import Path
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from preprocess import load_and_clean, fix_length, SR, N_SAMPLES, TRIM_DB  # noqa: E402

TR = {"angry": ("Öfke", "😠"), "happy": ("Mutluluk", "😄"), "sad": ("Üzüntü", "😢"),
      "fear": ("Korku", "😨"), "disgust": ("İğrenme", "🤢"), "neutral": ("Nötr", "😐")}
CLASSES = ["angry", "happy", "sad", "fear", "disgust", "neutral"]


def cli_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn", default="cnn_all/cnn_best.pt")
    ap.add_argument("--classic_dir", default="results_v3")
    ap.add_argument("--classic_model", default="svm", help="svm | rf | mlp | knn")
    TAG = {"rf": "random"}                                       # baseline.py RF'yi model_random.joblib diye kaydeder
    a, _ = ap.parse_known_args()
    a.classic_tag = TAG.get(a.classic_model, a.classic_model)
    return a


@st.cache_resource(show_spinner="CNN modeli yükleniyor...")
def get_cnn(path):
    from cnn import load_model
    model, ck = load_model(path, "cpu")
    return model, ck


@st.cache_resource(show_spinner="Klasik model yükleniyor...")
def get_classic(d, tag):
    import joblib
    d = Path(d)
    return joblib.load(d / "scaler.joblib"), joblib.load(d / f"model_{tag}.joblib")


def windows(y_clean_full):
    """Uzun kayıtları 3 sn pencere, 1,5 sn kaydırma ile böler (kısa kayıtlar tek pencere)."""
    if len(y_clean_full) <= int(3.5 * SR):
        return [fix_length(y_clean_full)]
    hop = N_SAMPLES // 2
    starts = range(0, len(y_clean_full) - N_SAMPLES + 1, hop)
    return [y_clean_full[s:s + N_SAMPLES] for s in starts]


def load_audio(raw_bytes, suffix=".wav"):
    """Dosyayı oku -> (pencereler, ek ölçümler, temizlenmiş tam sinyal)."""
    import librosa
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as f:
        f.write(raw_bytes); f.flush()
        y_fixed, extras = load_and_clean(f.name)           # eğitimle birebir aynı ön işleme (3 sn)
        y, _ = librosa.load(f.name, sr=SR, mono=True)
    if y_fixed is None:
        return None, None, None
    y, _ = librosa.effects.trim(y, top_db=TRIM_DB)
    y = (y / max(np.max(np.abs(y)), 1e-8)).astype(np.float32)
    wins = windows(y) if len(y) > int(3.5 * SR) else [y_fixed]
    return np.stack(wins).astype(np.float32), extras, y


def predict_cnn(model, W):
    import torch
    with torch.no_grad():
        return torch.softmax(model(torch.from_numpy(W)), 1).numpy().mean(0)


def predict_classic(scaler, model, W, extras):
    from features import handcrafted
    Fm = np.stack([np.concatenate([handcrafted(w), extras]) for w in W])
    if scaler.n_features_in_ == Fm.shape[1] - 4:                 # --ablate_extras ile eğitilmişse
        Fm = Fm[:, :-4]
    Z = scaler.transform(Fm)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(Z).mean(0), True
    s = model.decision_function(Z).mean(0)                       # SVM: kalibre edilmemiş puan
    e = np.exp(s - s.max())
    return e / e.sum(), False


def logmel_fig(y):
    import librosa, librosa.display
    import matplotlib.pyplot as plt
    S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=SR, n_fft=400, hop_length=160,
                                                           window="hamming", n_mels=128), ref=np.max)
    fig, ax = plt.subplots(figsize=(7, 2.6))
    librosa.display.specshow(S, sr=SR, hop_length=160, x_axis="time", y_axis="mel", ax=ax)
    ax.set_title("Log-mel spektrogram"); fig.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="Konuşmadan Duygu Tanıma", page_icon="🎙️", layout="centered")
    a = cli_args()
    st.title("🎙️ Konuşmadan Duygu Tanıma")
    st.caption("Fırat Üniversitesi Bilgisayar Mühendisliği · Takım Tertipler · Hafta 4 demo")

    with st.sidebar:
        st.header("Model")
        choices = []
        if Path(a.cnn).exists():
            choices.append("CNN (log-mel)")
        if (Path(a.classic_dir) / "scaler.joblib").exists():
            choices.append(f"Klasik ({a.classic_model.upper()})")
        if not choices:
            st.error(f"Model bulunamadı.\n\nCNN: `{a.cnn}`\n\nKlasik: `{a.classic_dir}`")
            st.stop()
        kind = st.radio("Kullanılacak model", choices)
        st.markdown("**Sınıflar:** " + ", ".join(f"{TR[c][1]} {TR[c][0]}" for c in CLASSES))
        st.info("Modeller İngilizce, oyuncular tarafından seslendirilmiş kayıtlarla (RAVDESS, CREMA-D) "
                "eğitildi. Doğal konuşmada ve farklı mikrofonlarda başarım düşebilir.")

    tab1, tab2 = st.tabs(["📁 Dosya yükle", "🎤 Mikrofon"])
    raw, suffix = None, ".wav"
    with tab1:
        up = st.file_uploader("Ses dosyası (wav, mp3, ogg, flac)", type=["wav", "mp3", "ogg", "flac", "m4a"])
        if up is not None:
            raw, suffix = up.getvalue(), Path(up.name).suffix or ".wav"
    with tab2:
        rec = st.audio_input("Kaydetmek için tıklayın, 2-5 sn konuşun")
        if rec is not None:
            raw = rec.getvalue()
    if raw is None:
        st.write("Bir kayıt yükleyin ya da mikrofonla kaydedin.")
        return

    st.audio(raw)
    t0 = time.time()
    W, extras, y_full = load_audio(raw, suffix)
    if W is None:
        st.error("Kayıtta konuşma bulunamadı (tamamen sessiz).")
        return
    calibrated = True
    if kind.startswith("CNN"):
        model, ck = get_cnn(a.cnn)
        probs = predict_cnn(model, W)
    else:
        scaler, model = get_classic(a.classic_dir, a.classic_tag)
        probs, calibrated = predict_classic(scaler, model, W, extras)
    dt = time.time() - t0

    top = int(np.argmax(probs))
    name, emo = TR[CLASSES[top]]
    c1, c2, c3 = st.columns(3)
    c1.metric("Tahmin", f"{emo} {name}")
    c2.metric("Güven", f"%{probs[top]*100:.0f}")
    c3.metric("Süre", f"{dt:.2f} sn")
    if len(W) > 1:
        st.caption(f"Kayıt {len(y_full)/SR:.1f} sn; {len(W)} pencerenin ortalaması alındı.")
    if not calibrated:
        st.caption("SVM olasılık üretmediği için karar puanlarının softmax'ı gösteriliyor (kalibre değil).")

    import pandas as pd
    df = pd.DataFrame({"Duygu": [f"{TR[c][1]} {TR[c][0]}" for c in CLASSES], "Olasılık": probs})
    st.bar_chart(df.set_index("Duygu"), horizontal=True)
    with st.expander("Log-mel spektrogram"):
        st.pyplot(logmel_fig(y_full))


main()
