"""Deteksi Penyakit Daun (v3) - demo Streamlit.

File yang harus ada di folder yang sama:
    leaf_model_v3.keras, classes_v3.json
"""
import json

import keras
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

st.set_page_config(page_title="Deteksi Penyakit Daun", page_icon="🌿")

SKALA = (1.0, 0.8, 0.6)   # tingkat zoom untuk cek konsistensi
THRESHOLD = 0.70


@st.cache_resource(show_spinner="Memuat model...")
def muat_model():
    model = keras.models.load_model("leaf_model_v3.keras", compile=False)
    with open("classes_v3.json") as f:
        class_names = json.load(f)
    return model, class_names


def bersihkan(nama):
    return nama.replace("___", " — ").replace("_", " ")


def prediksi(model, class_names, img):
    x = tf.convert_to_tensor(np.array(img), dtype=tf.float32)
    batch = []
    for k in SKALA:
        c = tf.image.central_crop(x, k) if k < 1.0 else x
        batch.append(tf.image.resize(c, (224, 224)))
    probs_all = model(tf.stack(batch), training=False).numpy()
    kelas = [int(np.argmax(p)) for p in probs_all]
    return probs_all[0], kelas   # tampilkan hasil dari foto utuh


model, class_names = muat_model()
IDX_BD = class_names.index("bukan_daun")

st.title("🌿 Deteksi Penyakit Daun")
st.write(
    "Klasifikasi 38 kelas daun (dataset PlantVillage) dengan MobileNetV2 + transfer learning. "
    "Model juga memeriksa konsistensi prediksinya di beberapa tingkat zoom."
)
st.warning(
    "**Keterbatasan:** model hanya mengenal 38 kelas tanaman/penyakit, dilatih pada foto "
    "berlatar polos, dan bisa keliru pada foto lapangan atau benda bertekstur mirip daun "
    "sakit (misalnya rumput kering). Bukan pengganti diagnosis ahli. Jangan unggah foto pribadi."
)

berkas = st.file_uploader("Unggah foto satu daun", type=["jpg", "jpeg", "png", "webp"])

if berkas is not None:
    img = ImageOps.exif_transpose(Image.open(berkas)).convert("RGB")
    img.thumbnail((1024, 1024))   # batasi ukuran agar hemat memori
    st.image(img, caption="Foto yang diunggah")

    with st.spinner("Menganalisis..."):
        probs, kelas = prediksi(model, class_names, img)

    top3 = np.argsort(probs)[-3:][::-1]
    st.subheader("Prediksi")
    for i in top3:
        st.write(f"**{bersihkan(class_names[i])}** — {probs[i]:.0%}")
        st.progress(float(probs[i]))

    if len(set(kelas)) > 1:
        st.warning("⚠️ Prediksi berubah-ubah saat difoto lebih dekat. Hasil tidak bisa "
                   "diandalkan. Coba foto ulang yang lebih jelas.")
    elif kelas[0] == IDX_BD:
        st.info("Ini sepertinya bukan daun dari tanaman yang dikenali model. "
                "Coba foto satu daun dengan jelas.")
    elif probs[top3[0]] < THRESHOLD:
        st.warning("⚠️ Model kurang yakin. Mungkin fotonya kurang jelas.")
    else:
        st.info("Model tidak berubah pikiran di beberapa tingkat zoom, tapi ini BUKAN "
                "jaminan benar. Konfirmasi ke ahli sebelum bertindak.")
