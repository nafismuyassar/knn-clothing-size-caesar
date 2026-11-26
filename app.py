"""
app.py — Streamlit App: Prediksi Ukuran Pakaian (KNN + CAESAR)
Nafis Muyassar
"""
import pickle

import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = "model/model.pkl"
SIZE_ORDER = ["XS", "S", "M", "L", "XL", "XXL"]

EN13402 = {
    "Pria": {
        "XS": "78–86 cm",  "S": "86–94 cm",   "M": "94–102 cm",
        "L": "102–110 cm", "XL": "110–118 cm", "XXL": "118–129 cm",
    },
    "Wanita": {
        "XS": "74–82 cm",  "S": "82–90 cm",   "M": "90–98 cm",
        "L": "98–107 cm",  "XL": "107–119 cm", "XXL": "119–131 cm",
    },
}


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def do_predict(artefak, gender_int, height, weight, chest, waist, hip):
    bmi = weight / (height / 100) ** 2
    x = np.array([[gender_int, height, weight, chest, waist, hip, bmi]], dtype=float)
    sc, knn = artefak["scaler"], artefak["knn"]
    x_s = sc.transform(x).copy()
    x_s[:, artefak["feats"].index(artefak["w_feat"])] *= artefak["w_val"]
    pid = knn.predict(x_s)[0]
    proba = knn.predict_proba(x_s)[0]
    size = artefak["id_to_size"][pid]
    conf = float(proba[pid]) * 100
    dist = {artefak["size_order"][i]: float(p) * 100 for i, p in enumerate(proba)}
    return size, conf, round(float(bmi), 1), dist


def bmi_category(bmi):
    if bmi < 18.5:
        return "Berat badan kurang"
    if bmi < 25:
        return "Berat badan normal"
    if bmi < 30:
        return "Kelebihan berat badan"
    return "Obesitas"


# ── Setup halaman ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Ukuran Pakaian",
    page_icon=None,
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    html, body, .stApp, .stApp p, .stApp label, .stApp li, .stApp button,
    .stApp input, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
    [data-testid="stCaptionContainer"], [data-testid="stTable"] {
        font-family: 'IBM Plex Sans', 'Segoe UI', Arial, sans-serif;
    }
    .block-container { max-width: 1100px; padding-top: 2.5rem; padding-bottom: 3rem; }
    h1 { font-weight: 600; letter-spacing: -0.01em; font-size: 1.9rem; }
    h2, h3 { font-weight: 600; font-size: 1.15rem; }
    [data-testid="stMetricValue"] { font-weight: 600; }
    #MainMenu, footer, header [data-testid="stToolbar"] { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    artefak = load_model()
except FileNotFoundError:
    st.error("Model tidak ditemukan. Jalankan `python train_model.py` terlebih dahulu.")
    st.stop()

# ── Header ───────────────────────────────────────────────────────────────────
st.title("Prediksi Ukuran Pakaian")
st.caption(
    "Masukkan data tubuh Anda untuk mendapatkan rekomendasi ukuran "
    "berdasarkan standar EN 13402."
)

left, right = st.columns([5, 6], gap="large")

# ── Kolom kiri: input ────────────────────────────────────────────────────────
with left:
    with st.container(border=True):
        st.subheader("Data tubuh")

        gender = st.radio("Jenis kelamin", ["Pria", "Wanita"], horizontal=True)

        c1, c2 = st.columns(2)
        with c1:
            height = st.number_input("Tinggi badan (cm)", 100.0, 250.0, 170.0, 0.5)
        with c2:
            weight = st.number_input("Berat badan (kg)", 30.0, 200.0, 50.0, 0.5)

        bmi_now = weight / (height / 100) ** 2
        st.caption(f"BMI {bmi_now:.1f} — {bmi_category(bmi_now)}")

        st.markdown("**Lingkar tubuh**")
        chest = st.number_input(
            "Lingkar dada (cm)", 50.0, 200.0, 96.0, 0.5,
            help="Ukur di bagian terlebar dada, melingkar horizontal.",
        )
        c3, c4 = st.columns(2)
        with c3:
            waist = st.number_input("Lingkar pinggang (cm)", 40.0, 200.0, 90.0, 0.5)
        with c4:
            hip = st.number_input("Lingkar pinggul (cm)", 50.0, 200.0, 100.0, 0.5)

        submitted = st.button("Prediksi ukuran", type="primary", use_container_width=True)

# ── Kolom kanan: hasil ───────────────────────────────────────────────────────
with right:
    if not submitted:
        with st.container(border=True):
            st.subheader("Hasil prediksi")
            st.write("Isi data di sebelah kiri, lalu tekan **Prediksi ukuran**.")
            st.caption(
                "Prediksi memakai data CAESAR dan standar EN 13402. Ukuran aktual "
                "tiap merek bisa berbeda, jadi gunakan hasilnya sebagai acuan."
            )
    else:
        gender_int = 0 if gender == "Pria" else 1
        size, conf, bmi_val, dist = do_predict(
            artefak, gender_int, height, weight, chest, waist, hip
        )

        with st.container(border=True):
            st.subheader("Hasil prediksi")
            m1, m2 = st.columns(2)
            m1.metric("Ukuran yang disarankan", size)
            m2.metric("Tingkat keyakinan", f"{conf:.1f}%")

            st.markdown("**Distribusi tetangga terdekat**")
            k = artefak.get("k", "?")
            st.caption(f"Proporsi ukuran dari {k} data terdekat")
            for s in SIZE_ORDER:
                pct = dist.get(s, 0)
                suffix = " (prediksi)" if s == size else ""
                st.progress(min(int(round(pct)), 100), text=f"{s}{suffix}: {pct:.1f}%")

        with st.container(border=True):
            st.subheader(f"Rentang EN 13402 untuk {gender}")
            col_header = "Lingkar dada" if gender == "Pria" else "Bust"
            tbl = EN13402[gender]
            df = pd.DataFrame(
                {
                    "Ukuran": SIZE_ORDER,
                    col_header: [tbl[s] for s in SIZE_ORDER],
                }
            )

            def highlight(row):
                if row["Ukuran"] == size:
                    return ["background-color: #DCE8F5; font-weight: 600"] * len(row)
                return [""] * len(row)

            st.dataframe(
                df.style.apply(highlight, axis=1),
                hide_index=True,
                use_container_width=True,
            )

        st.caption(
            "Prediksi berdasarkan data CAESAR (Amerika dan Eropa) serta standar "
            "EN 13402. Ukuran aktual tiap merek pakaian bisa berbeda."
        )

# ── Info model ───────────────────────────────────────────────────────────────
st.write("")
with st.expander("Tentang model"):
    k = artefak.get("k", "?")
    wv = artefak.get("w_val", "?")
    acc = artefak.get("holdout_acc", 0) * 100
    f1 = artefak.get("holdout_f1", 0) * 100
    st.markdown(
        f"""
| Parameter | Nilai |
|-----------|-------|
| Algoritma | K-Nearest Neighbor (KNN) |
| k (tetangga) | {k} |
| Bobot lingkar dada | ×{wv} |
| Jarak | Euclidean |
| Scaler | MinMaxScaler |
| Dataset | CAESAR (4.435 sampel, 7 fitur) |
| Akurasi hold-out | {acc:.2f}% |
| F1-macro hold-out | {f1:.2f}% |
"""
    )

st.divider()
st.caption(
    "Prediksi Ukuran Pakaian Menggunakan Algoritma K-Nearest Neighbor dengan "
    "Pembobotan Fitur Lingkar Dada pada Dataset CAESAR — Nafis Muyassar"
)
