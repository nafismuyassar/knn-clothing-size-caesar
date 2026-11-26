"""
app.py — Streamlit App: Prediksi Ukuran Pakaian (KNN + CAESAR)
Nafis Muyassar
"""
import pickle
import numpy as np
import streamlit as st

MODEL_PATH = "model/model.pkl"
SIZE_ORDER = ["XS", "S", "M", "L", "XL", "XXL"]

EN13402 = {
    "Pria": {
        "XS": "78–86 cm",   "S": "86–94 cm",   "M":  "94–102 cm",
        "L": "102–110 cm",  "XL": "110–118 cm", "XXL": "118–129 cm",
    },
    "Wanita": {
        "XS": "74–82 cm",   "S": "82–90 cm",   "M":  "90–98 cm",
        "L":  "98–107 cm",  "XL": "107–119 cm", "XXL": "119–131 cm",
    },
}


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def do_predict(artefak, gender_int, height, weight, chest, waist, hip):
    bmi = weight / (height / 100) ** 2
    x   = np.array([[gender_int, height, weight, chest, waist, hip, bmi]], dtype=float)
    sc, knn = artefak["scaler"], artefak["knn"]
    x_s = sc.transform(x).copy()
    x_s[:, artefak["feats"].index(artefak["w_feat"])] *= artefak["w_val"]
    pid   = knn.predict(x_s)[0]
    proba = knn.predict_proba(x_s)[0]
    size  = artefak["id_to_size"][pid]
    conf  = float(proba[pid]) * 100
    dist  = {artefak["size_order"][i]: float(p) * 100 for i, p in enumerate(proba)}
    return size, conf, round(float(bmi), 1), dist


def bmi_category(bmi):
    if bmi < 18.5: return "Berat badan kurang"
    if bmi < 25:   return "Berat badan normal"
    if bmi < 30:   return "Kelebihan berat badan"
    return "Obesitas"


# ── Setup ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FitSize — Prediksi Ukuran Pakaian",
    page_icon="👕",
    layout="centered",
)

try:
    artefak = load_model()
except FileNotFoundError:
    st.error("Model tidak ditemukan. Jalankan `python train_model.py` dulu.")
    st.stop()

# ── Header ───────────────────────────────────────────────────────────────────
st.title("👕 FitSize")
st.caption("Prediksi ukuran pakaian berdasarkan ukuran tubuh & standar EN 13402")
st.divider()

# ── Form input ───────────────────────────────────────────────────────────────
st.subheader("Data Tubuh")

gender = st.radio("Jenis Kelamin", ["Pria", "Wanita"], horizontal=True)

col1, col2 = st.columns(2)
with col1:
    height = st.number_input("Tinggi Badan (cm)",       min_value=100.0, max_value=250.0, value=170.0, step=0.5)
    weight = st.number_input("Berat Badan (kg)",         min_value=30.0,  max_value=200.0, value=50.0,  step=0.5)
    chest  = st.number_input("Lingkar Dada / Bust (cm)", min_value=50.0,  max_value=200.0, value=96.0,  step=0.5)
with col2:
    waist  = st.number_input("Lingkar Pinggang (cm)",    min_value=40.0,  max_value=200.0, value=90.0,  step=0.5)
    hip    = st.number_input("Lingkar Pinggul (cm)",     min_value=50.0,  max_value=200.0, value=100.0, step=0.5)
    bmi_now = weight / (height / 100) ** 2
    st.metric("BMI (otomatis)", f"{bmi_now:.1f}")
    st.caption(bmi_category(bmi_now))

st.divider()

# ── Prediksi ─────────────────────────────────────────────────────────────────
if st.button("🔍 Prediksi Ukuran", type="primary", use_container_width=True):
    gender_int = 0 if gender == "Pria" else 1
    size, conf, bmi_val, dist = do_predict(
        artefak, gender_int, height, weight, chest, waist, hip)

    # Hasil utama
    st.subheader("Hasil Prediksi")
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.metric("Ukuran", size)
        st.caption(f"Kepercayaan: **{conf:.1f}%**")
        st.caption(f"Standar EN 13402 · {gender}")
    with col_b:
        st.caption("Distribusi tetangga terdekat (k=15)")
        for s in SIZE_ORDER:
            pct = dist.get(s, 0)
            label = f"**{s}** ← Prediksi" if s == size else s
            st.progress(int(pct), text=f"{label}  {pct:.1f}%")

    st.divider()

    # Tabel EN 13402
    st.subheader(f"Rentang EN 13402 — {gender}")
    tbl = EN13402[gender]
    col_header = "Lingkar Dada" if gender == "Pria" else "Bust"
    rows = []
    for s in SIZE_ORDER:
        mark = "✅ Kamu" if s == size else ""
        rows.append({"Ukuran": s, col_header: tbl[s], "": mark})
    st.table(rows)

    # Catatan
    st.info(
        "Prediksi berdasarkan data CAESAR (Amerika/Eropa) & standar EN 13402. "
        "Ukuran aktual tiap merek pakaian bisa berbeda — gunakan sebagai referensi."
    )

# ── Info model ───────────────────────────────────────────────────────────────
with st.expander("ℹ️ Tentang Model"):
    k   = artefak.get("k", "?")
    wv  = artefak.get("w_val", "?")
    acc = artefak.get("holdout_acc", 0) * 100
    f1  = artefak.get("holdout_f1",  0) * 100
    st.markdown(f"""
| Parameter | Nilai |
|-----------|-------|
| Algoritma | K-Nearest Neighbor (KNN) |
| k (tetangga) | {k} |
| Bobot lingkar dada | ×{wv} |
| Jarak | Euclidean |
| Scaler | MinMaxScaler |
| Dataset | CAESAR (4.435 sampel, 7 fitur) |
| Hold-out Accuracy | **{acc:.2f}%** |
| Hold-out F1-macro | **{f1:.2f}%** |
""")

# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "*Prediksi Ukuran Pakaian Menggunakan Algoritma K-Nearest Neighbor "
    "dengan Pembobotan Fitur Lingkar Dada pada Dataset CAESAR* · "
)
