"""
app.py — Streamlit App: Prediksi Ukuran Pakaian (KNN + CAESAR)
"""
import pickle
import numpy as np
import streamlit as st

# ── Konstanta ────────────────────────────────────────────────────────────────
MODEL_PATH = "model/model.pkl"

SIZE_COLOR = {
    "XS":  "#6BAED6",
    "S":   "#74C476",
    "M":   "#FD8D3C",
    "L":   "#9E9AC8",
    "XL":  "#FC4E2A",
    "XXL": "#BD0026",
}

EN13402_TABLE = {
    "Pria (chest girth)": {
        "XS":  "78–86 cm",  "S": "86–94 cm",   "M":  "94–102 cm",
        "L":  "102–110 cm", "XL":"110–118 cm",  "XXL":"118–129 cm",
    },
    "Wanita (bust girth)": {
        "XS":  "74–82 cm",  "S": "82–90 cm",   "M":  "90–98 cm",
        "L":   "98–107 cm", "XL":"107–119 cm",  "XXL":"119–131 cm",
    },
}

# ── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ── Prediksi ─────────────────────────────────────────────────────────────────
def predict(artefak, gender_int, height_cm, weight_kg, chest_cm,
            waist_cm, hip_cm):
    bmi = weight_kg / (height_cm / 100) ** 2
    x = np.array([[gender_int, height_cm, weight_kg, chest_cm,
                   waist_cm, hip_cm, bmi]], dtype=float)

    sc  = artefak["scaler"]
    knn = artefak["knn"]
    feats  = artefak["feats"]
    w_feat = artefak["w_feat"]
    w_val  = artefak["w_val"]
    id_to_size = artefak["id_to_size"]

    x_s = sc.transform(x).copy()
    j   = feats.index(w_feat)
    x_s[:, j] *= w_val

    pred_id    = knn.predict(x_s)[0]
    proba      = knn.predict_proba(x_s)[0]        # proporsi k tetangga
    size_order = artefak["size_order"]

    size_label = id_to_size[pred_id]
    confidence = float(proba[pred_id]) * 100
    bmi_val    = round(float(bmi), 1)

    # Distribusi semua ukuran (untuk bar chart)
    dist = {size_order[i]: float(p) * 100 for i, p in enumerate(proba)}
    return size_label, confidence, bmi_val, dist


# ── Layout Streamlit ─────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Prediksi Ukuran Pakaian",
        page_icon="👕",
        layout="centered",
    )

    # ── Header ──
    st.title("👕 Prediksi Ukuran Pakaian")
    st.markdown(
        "Masukkan data tubuh di bawah. Model **K-Nearest Neighbor** "
        "akan memprediksi ukuran pakaian berdasarkan standar **EN 13402**."
    )
    st.divider()

    # ── Load model ──
    try:
        artefak = load_model()
    except FileNotFoundError:
        st.error(
            "⚠️ File model tidak ditemukan (`model/model.pkl`). "
            "Jalankan `python train_model.py` terlebih dahulu."
        )
        st.stop()

    # ── Info model ──
    with st.expander("ℹ️ Tentang Model", expanded=False):
        k    = artefak.get("k", "?")
        w    = artefak.get("w_val", "?")
        acc  = artefak.get("holdout_acc", 0) * 100
        f1   = artefak.get("holdout_f1", 0) * 100
        st.markdown(f"""
| Parameter | Nilai |
|-----------|-------|
| Algoritma | K-Nearest Neighbor (KNN) |
| k (tetangga) | {k} |
| Bobot lingkar dada | ×{w} |
| Jarak | Euclidean |
| Scaler | MinMaxScaler |
| Dataset | CAESAR ({4435} sampel, 7 fitur) |
| Hold-out Accuracy | {acc:.2f}% |
| Hold-out F1-macro | {f1:.2f}% |
        """)
        st.caption(
            "Model dilatih dengan nested cross-validation. "
            "Label ukuran mengacu pada standar EN 13402."
        )

    # ── Form input ──
    st.subheader("📋 Data Tubuh")
    col1, col2 = st.columns(2)

    with col1:
        gender = st.radio("Jenis Kelamin", ["Pria", "Wanita"], horizontal=True)
        height = st.number_input("Tinggi Badan (cm)", min_value=100.0,
                                 max_value=250.0, value=170.0, step=0.5)
        weight = st.number_input("Berat Badan (kg)", min_value=30.0,
                                 max_value=200.0, value=50.0, step=0.5)
        chest  = st.number_input("Lingkar Dada / Bust (cm)", min_value=50.0,
                                 max_value=200.0, value=96.0, step=0.5)

    with col2:
        waist  = st.number_input("Lingkar Pinggang (cm)", min_value=40.0,
                                 max_value=200.0, value=90.0, step=0.5)
        hip    = st.number_input("Lingkar Pinggul (cm)", min_value=50.0,
                                 max_value=200.0, value=100.0, step=0.5)
        bmi_display = weight / (height / 100) ** 2
        st.metric("BMI (otomatis)", f"{bmi_display:.1f}")
        st.caption(_bmi_category(bmi_display))

    # ── Tombol prediksi ──
    st.divider()
    if st.button("🔍 Prediksi Ukuran", type="primary", use_container_width=True):
        gender_int = 0 if gender == "Pria" else 1
        size, conf, bmi_val, dist = predict(
            artefak, gender_int, height, weight, chest, waist, hip)

        # ── Hasil utama ──
        color = SIZE_COLOR.get(size, "#555")
        st.markdown(f"""
<div style="text-align:center; padding: 1.5rem 1rem;
     background:{color}22; border:2px solid {color};
     border-radius:12px; margin-bottom:1rem;">
  <span style="font-size:3.5rem; font-weight:bold; color:{color};">{size}</span><br>
  <span style="font-size:1.1rem; color:#555;">
    Kepercayaan: <strong>{conf:.1f}%</strong> dari {artefak["k"]} tetangga
  </span>
</div>
""", unsafe_allow_html=True)

        # ── Distribusi ukuran ──
        st.subheader("📊 Distribusi Tetangga per Ukuran")
        size_order = artefak["size_order"]
        vals = [dist.get(s, 0) for s in size_order]
        bar_html = _bar_chart_html(size_order, vals, SIZE_COLOR)
        st.markdown(bar_html, unsafe_allow_html=True)

        # ── Rentang lingkar EN 13402 ──
        st.subheader("📏 Rentang EN 13402")
        gender_key = "Pria (chest girth)" if gender == "Pria" else "Wanita (bust girth)"
        tbl = EN13402_TABLE[gender_key]
        md = "| Ukuran | Rentang Lingkar |  |\n|--------|----------------|---|\n"
        for s in size_order:
            mark = "✅ **Anda**" if s == size else ""
            md  += f"| **{s}** | {tbl.get(s, '-')} | {mark} |\n"
        st.markdown(md)

        # ── Disclaimer ──
        st.info(
            "ℹ️ **Catatan:** Prediksi berdasarkan data CAESAR (populasi Amerika/Eropa). "
            "Hasilnya bersifat referensi — cocokan dengan ukuran aktual merek pakaian "
            "yang Anda pakai, karena tiap merek bisa berbeda."
        )

    # ── Footer ──
    st.divider()
    st.caption(
        "*Prediksi Ukuran Pakaian Menggunakan Algoritma K-Nearest Neighbor "
        "dengan Pembobotan Fitur Lingkar Dada pada Dataset CAESAR* · "
    )


# ── Helper ───────────────────────────────────────────────────────────────────
def _bmi_category(bmi):
    if bmi < 18.5:
        return "🔵 Berat badan kurang"
    elif bmi < 25:
        return "🟢 Berat badan normal"
    elif bmi < 30:
        return "🟡 Kelebihan berat badan"
    else:
        return "🔴 Obesitas"


def _bar_chart_html(labels, values, color_map):
    bars = ""
    for lbl, val in zip(labels, values):
        col = color_map.get(lbl, "#888")
        bars += f"""
<div style="display:flex; align-items:center; margin-bottom:6px; gap:8px;">
  <span style="width:36px; font-weight:bold; color:{col};">{lbl}</span>
  <div style="flex:1; background:#eee; border-radius:4px; height:20px;">
    <div style="width:{val:.1f}%; background:{col}; height:100%;
                border-radius:4px; min-width:2px;"></div>
  </div>
  <span style="width:46px; font-size:0.85rem; color:#555;">{val:.1f}%</span>
</div>"""
    return f'<div style="padding:0.5rem 0;">{bars}</div>'


if __name__ == "__main__":
    main()
