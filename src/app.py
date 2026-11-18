"""
Konuşmadan Duygu Tanıma - Demo arayüzü (H6)  |  Takım: Tertipler
Fırat Üniversitesi · Bilgisayar Mühendisliği Tasarım Dersi · Hafta 6 (sunum sürümü)

Ses dosyası yükleyerek ya da mikrofonla konuşarak 6 duygudan birini tahmin eder ve
her sınıfın olasılığını gösterir. Ses, eğitimdeki ile AYNI ön işlemeden geçer
(preprocess.load_and_clean: 16 kHz mono, sessizlik kesme, genlik normalizasyonu, 3 sn).
3,5 sn'den uzun kayıtlar 3 sn'lik örtüşen pencerelere bölünür, olasılıkları ortalanır.

Desteklenen modeller (hangisinin dosyası varsa otomatik görünür):
  wav2vec 2.0 + SVM : en iyi model. results_w2v/{scaler,model_svm}.joblib + katman no
                      (features_v3/extra_features_info.json) + wav2vec2-base kodlayıcı.
  CNN (log-mel)     : cnn.py çıktısı (ör. cnn_all/cnn_best.pt)
  Klasik (556)      : baseline.py çıktısı (results_v3/scaler.joblib + model_*.joblib)

Çalıştırma (örnek):
  streamlit run src/app.py -- \
     --w2v_dir results_w2v --feat_info features_v3/extra_features_info.json \
     --cnn cnn_all/cnn_best.pt --classic_dir results_v3
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from preprocess import load_and_clean, fix_length, SR, N_SAMPLES, TRIM_DB  # noqa: E402

# ----------------------------------------------------------------------------- sabitler
CLASSES = ["angry", "happy", "sad", "fear", "disgust", "neutral"]
TR = {
    "angry":   ("Öfke",     "😠", "#e4572e"),
    "happy":   ("Mutluluk", "😄", "#f3a712"),
    "sad":     ("Üzüntü",   "😢", "#4a8fe7"),
    "fear":    ("Korku",    "😨", "#8e6cfc"),
    "disgust": ("İğrenme",  "🤢", "#4caf7d"),
    "neutral": ("Nötr",     "😐", "#8a94a6"),
}
# arousal (uyarılmışlık) / valence (değerlik): sunumdaki duygu haritası için kaba konum (-1..1)
AV = {"angry": (0.8, -0.7), "happy": (0.8, 0.8), "sad": (-0.7, -0.6),
      "fear": (0.7, -0.8), "disgust": (0.2, -0.7), "neutral": (0.0, 0.0)}

W2V_MODEL_DEFAULT = "facebook/wav2vec2-base"
W2V_HOP = 320  # wav2vec 2.0: ~20 ms / çerçeve


# ----------------------------------------------------------------------------- CLI
def cli_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w2v_dir", default="results_w2v",
                    help="wav2vec+SVM klasörü (scaler.joblib + model_svm.joblib)")
    ap.add_argument("--feat_info", default="features_v3/extra_features_info.json",
                    help="seçilen wav2vec katmanını içeren json")
    ap.add_argument("--w2v_model", default=W2V_MODEL_DEFAULT)
    ap.add_argument("--cnn", default="cnn_all/cnn_best.pt")
    ap.add_argument("--classic_dir", default="results_v3")
    ap.add_argument("--classic_model", default="svm", help="svm | rf | mlp | knn")
    a, _ = ap.parse_known_args()
    a.classic_tag = {"rf": "random"}.get(a.classic_model, a.classic_model)
    return a


# ----------------------------------------------------------------------------- model yükleme (önbellekli)
@st.cache_resource(show_spinner="wav2vec 2.0 kodlayıcı yükleniyor (ilk seferde biraz sürer)...")
def get_w2v_encoder(model_name):
    import torch
    from transformers import AutoFeatureExtractor, AutoModel
    fe = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True).eval()
    torch.set_grad_enabled(False)
    return fe, model


@st.cache_resource(show_spinner="wav2vec+SVM yükleniyor...")
def get_w2v_head(d, info_path):
    import joblib
    d = Path(d)
    scaler = joblib.load(d / "scaler.joblib")
    model = joblib.load(d / "model_svm.joblib")
    layer = 5
    try:
        layer = int(json.load(open(info_path))["w2v"]["layer"])
    except Exception:
        pass
    return scaler, model, layer


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


# ----------------------------------------------------------------------------- ses
def windows(y):
    """Uzun kayıtları 3 sn pencere / 1,5 sn kaydırma ile böler (kısa kayıt tek pencere)."""
    if len(y) <= int(3.5 * SR):
        return [fix_length(y)]
    hop = N_SAMPLES // 2
    starts = list(range(0, len(y) - N_SAMPLES + 1, hop))
    wins = [y[s:s + N_SAMPLES] for s in starts]
    if starts and starts[-1] + N_SAMPLES < len(y):          # son pencereyi de kapsa
        wins.append(y[-N_SAMPLES:])
    return wins


def load_audio(raw_bytes, suffix=".wav"):
    """Dosyayı oku -> (pencereler (N,L), ek ölçümler, temizlenmiş tam sinyal)."""
    import librosa
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as f:
        f.write(raw_bytes)
        f.flush()
        y_fixed, extras = load_and_clean(f.name)            # eğitimle birebir aynı ön işleme
        y, _ = librosa.load(f.name, sr=SR, mono=True)
    if y_fixed is None:
        return None, None, None
    y, _ = librosa.effects.trim(y, top_db=TRIM_DB)
    y = (y / max(np.max(np.abs(y)), 1e-8)).astype(np.float32)
    wins = windows(y) if len(y) > int(3.5 * SR) else [y_fixed]
    return np.stack(wins).astype(np.float32), extras, y


# ----------------------------------------------------------------------------- tahmin
def _softmax(s):
    e = np.exp(s - s.max())
    return e / e.sum()


def predict_w2v(fe, encoder, scaler, svm, layer, W):
    """Her pencereyi wav2vec'ten geçir, seçilen katmanı zaman-ortala, SVM ile tahmin et.
    Sondaki sıfır dolgusu, geçerli çerçeve sayısı üzerinden ortalamadan dışlanır
    (extra_features.py ile tutarlı)."""
    embs = []
    for w in W:
        nz = np.flatnonzero(w)
        real_len = int(nz[-1] + 1) if nz.size else len(w)       # dolgu öncesi gerçek uzunluk
        inp = fe([w], sampling_rate=SR, return_tensors="pt", padding=True)
        hs = encoder(inp["input_values"]).hidden_states         # (L+1) x (1,T,D)
        h = hs[layer][0]                                         # (T,D)
        n_frames = max(1, min(h.shape[0], int(np.ceil(real_len / W2V_HOP))))
        embs.append(h[:n_frames].mean(0).numpy())
    Z = scaler.transform(np.stack(embs).astype(np.float32))
    if hasattr(svm, "predict_proba"):
        return svm.predict_proba(Z).mean(0), True
    return _softmax(svm.decision_function(Z).mean(0)), False


def predict_cnn(model, W):
    import torch
    with torch.no_grad():
        return torch.softmax(model(torch.from_numpy(W)), 1).numpy().mean(0)


def predict_classic(scaler, model, W, extras):
    from features import handcrafted
    Fm = np.stack([np.concatenate([handcrafted(w), extras]) for w in W])
    if scaler.n_features_in_ == Fm.shape[1] - 4:            # --ablate_extras ile eğitilmişse
        Fm = Fm[:, :-4]
    Z = scaler.transform(Fm)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(Z).mean(0), True
    return _softmax(model.decision_function(Z).mean(0)), False


# ----------------------------------------------------------------------------- görseller
def logmel_fig(y):
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    S = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=SR, n_fft=400, hop_length=160,
                                       window="hamming", n_mels=128), ref=np.max)
    fig, ax = plt.subplots(figsize=(7, 2.6))
    librosa.display.specshow(S, sr=SR, hop_length=160, x_axis="time", y_axis="mel", ax=ax)
    ax.set_title("Log-mel spektrogram")
    fig.tight_layout()
    return fig


def av_fig(cls):
    """Arousal-valence duygu haritasında tahmini işaretle (sunum için)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    ax.axhline(0, color="#ccc", lw=1)
    ax.axvline(0, color="#ccc", lw=1)
    for c, (a, v) in AV.items():
        name, emo, col = TR[c]
        big = (c == cls)
        ax.scatter(v, a, s=520 if big else 140, c=col, alpha=0.9 if big else 0.35,
                   edgecolors="black" if big else "none", zorder=3 if big else 2)
        ax.annotate(name, (v, a), ha="center", va="center",
                    fontsize=9 if big else 7, fontweight="bold" if big else "normal",
                    color="white" if big else "#333")
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("Değerlik (valence) →"); ax.set_ylabel("Uyarılmışlık (arousal) →")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    return fig


def prob_bars_html(probs):
    rows = []
    order = np.argsort(probs)[::-1]
    for i in order:
        c = CLASSES[i]
        name, emo, col = TR[c]
        pct = probs[i] * 100
        rows.append(
            f'<div class="barrow"><div class="barlbl">{emo} {name}</div>'
            f'<div class="bartrack"><div class="barfill" style="width:{pct:.1f}%;background:{col}"></div></div>'
            f'<div class="barval">%{pct:.0f}</div></div>')
    return '<div class="bars">' + "".join(rows) + "</div>"


CSS = """
<style>
.block-container{max-width:900px;padding-top:1.2rem}
.hero{border-radius:18px;padding:22px 26px;color:#fff;margin:6px 0 14px;
  box-shadow:0 10px 30px rgba(0,0,0,.18)}
.hero .emo{font-size:60px;line-height:1}
.hero .lbl{font-size:30px;font-weight:800;letter-spacing:.3px}
.hero .sub{opacity:.92;font-size:14px;margin-top:2px}
.bars{display:flex;flex-direction:column;gap:9px;margin-top:6px}
.barrow{display:grid;grid-template-columns:120px 1fr 48px;align-items:center;gap:10px}
.barlbl{font-size:14px;white-space:nowrap}
.bartrack{background:#eef0f4;border-radius:8px;height:16px;overflow:hidden}
.barfill{height:100%;border-radius:8px;transition:width .4s}
.barval{font-variant-numeric:tabular-nums;text-align:right;font-weight:600;font-size:13px}
</style>
"""


# ----------------------------------------------------------------------------- uygulama
def main():
    st.set_page_config(page_title="Konuşmadan Duygu Tanıma", page_icon="🎙️", layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)
    a = cli_args()

    st.title("🎙️ Konuşmadan Duygu Tanıma")
    st.caption("Fırat Üniversitesi · Bilgisayar Mühendisliği Tasarım Dersi · Takım **Tertipler**")

    # --- kullanılabilir modelleri tespit et
    available = {}
    if (Path(a.w2v_dir) / "scaler.joblib").exists() and (Path(a.w2v_dir) / "model_svm.joblib").exists():
        available["wav2vec 2.0 + SVM  ★ en iyi"] = "w2v"
    if Path(a.cnn).exists():
        available["CNN (log-mel)"] = "cnn"
    if (Path(a.classic_dir) / "scaler.joblib").exists():
        available[f"Klasik — {a.classic_model.upper()} (556 öznitelik)"] = "classic"

    with st.sidebar:
        st.header("⚙️ Model")
        if not available:
            st.error("Hiç model dosyası bulunamadı.\n\n"
                     f"wav2vec+SVM: `{a.w2v_dir}`\n\nCNN: `{a.cnn}`\n\nKlasik: `{a.classic_dir}`\n\n"
                     "Doğru klasörleri `--w2v_dir / --cnn / --classic_dir` ile verin.")
            st.stop()
        label = st.radio("Kullanılacak model", list(available.keys()))
        kind = available[label]
        st.markdown("**Sınıflar**")
        st.markdown("  ".join(f"{TR[c][1]} {TR[c][0]}" for c in CLASSES))
        st.info("Modeller İngilizce, oyuncular tarafından seslendirilmiş kayıtlarla (RAVDESS, "
                "CREMA-D) eğitildi. Doğal konuşmada ve farklı mikrofonlarda başarım düşebilir.")
        st.caption("En iyi model wav2vec 2.0 gömmeleri + SVM: birleşik test ≈ %70 doğruluk.")

    # --- giriş
    tab1, tab2 = st.tabs(["📁 Dosya yükle", "🎤 Mikrofon"])
    raw, suffix = None, ".wav"
    with tab1:
        up = st.file_uploader("Ses dosyası (wav, mp3, ogg, flac, m4a)",
                              type=["wav", "mp3", "ogg", "flac", "m4a"])
        if up is not None:
            raw, suffix = up.getvalue(), Path(up.name).suffix or ".wav"
    with tab2:
        rec = st.audio_input("Kaydetmek için tıklayın, 2–5 sn konuşun")
        if rec is not None:
            raw = rec.getvalue()

    if raw is None:
        st.write("👈 Soldan bir model seçin, sonra bir kayıt yükleyin ya da mikrofonla kaydedin.")
        return

    st.audio(raw)
    t0 = time.time()
    W, extras, y_full = load_audio(raw, suffix)
    if W is None:
        st.error("Kayıtta konuşma bulunamadı (kayıt tamamen sessiz).")
        return

    calibrated = True
    if kind == "w2v":
        fe, encoder = get_w2v_encoder(a.w2v_model)
        scaler, svm, layer = get_w2v_head(a.w2v_dir, a.feat_info)
        probs, calibrated = predict_w2v(fe, encoder, scaler, svm, layer, W)
    elif kind == "cnn":
        model, _ = get_cnn(a.cnn)
        probs = predict_cnn(model, W)
    else:
        scaler, model = get_classic(a.classic_dir, a.classic_tag)
        probs, calibrated = predict_classic(scaler, model, W, extras)
    dt = time.time() - t0

    top = int(np.argmax(probs))
    name, emo, col = TR[CLASSES[top]]

    # --- hero sonuç kartı
    st.markdown(
        f'<div class="hero" style="background:linear-gradient(135deg,{col},{col}cc)">'
        f'<span class="emo">{emo}</span>'
        f'<div class="lbl">{name}</div>'
        f'<div class="sub">güven %{probs[top]*100:.0f} · {dt:.2f} sn · model: {label}</div>'
        f'</div>', unsafe_allow_html=True)

    if len(W) > 1:
        st.caption(f"Kayıt {len(y_full)/SR:.1f} sn; {len(W)} pencerenin olasılıkları ortalandı.")
    if not calibrated:
        st.caption("Not: SVM kalibre edilmiş olasılık üretmez; karar puanlarının softmax'ı gösteriliyor.")

    # --- olasılık çubukları
    st.markdown("#### Sınıf olasılıkları")
    st.markdown(prob_bars_html(probs), unsafe_allow_html=True)

    # --- ek görseller
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("###### Duygu haritası")
        st.pyplot(av_fig(CLASSES[top]))
    with c2:
        st.markdown("###### Log-mel spektrogram")
        st.pyplot(logmel_fig(y_full))


main()
