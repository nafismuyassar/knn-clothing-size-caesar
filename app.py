"""
app.py — Streamlit App: Prediksi Ukuran Pakaian (KNN + CAESAR)
Nafis Muyassar
"""
import pickle
from pathlib import Path
import numpy as np
import streamlit as st

# Menggunakan Path agar aman saat deploy di Streamlit Cloud
MODEL_PATH = Path(__file__).parent / "model" / "model.pkl"
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
    page_icon="📏",
    layout="centered",
)

# ── CSS INJECTION (MODERN & CLEAN) ───────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --primary: #2563EB;
    --primary-hover: #1D4ED8;
    --bg-color: #F8FAFC;
    --text-main: #1E293B;
    --text-muted: #64748B;
    --border: #E2E8F0;
}

/* Global Font & Background */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-main);
}
.stApp {
    background-color: var(--bg-color) !important;
}

/* Hide Streamlit Header/Footer */
#MainMenu, footer, header, [data-testid="stHeader"] { visibility: hidden !important; }

/* Container width */
.block-container { max-width: 720px !important; padding-top: 3rem !important; }

/* Typography */
h1, h2, h3 { font-weight: 600 !important; color: var(--text-main) !important; letter-spacing: -0.02em !important; }

/* Custom Header */
.app-header { text-align: center; margin-bottom: 2rem; }
.app-header h1 { font-size: 2.5rem !important; color: var(--text-main) !important; margin-bottom: 0.5rem !important; font-weight: 700 !important; }
.app-header p { color: var(--text-muted); font-size: 1rem; margin: 0; }

/* Form Card Container */
.form-card {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    margin-bottom: 1.5rem;
}

/* Input Styling (Fix Dark Mode Bug) */
[data-testid="stNumberInput"] label { font-weight: 500 !important; font-size: 0.85rem !important; color: var(--text-muted) !important; }
[data-testid="stNumberInput"] input {
    border-radius: 6px !important;
    border: 1px solid var(--border) !important;
    padding: 0.5rem 0.75rem !important;
    background-color: #F8FAFC !important;
    color: var(--text-main) !important;
}
[data-testid="stNumberInput"] input:focus { border-color: var(--primary) !important; background-color: #FFFFFF !important; }

/* Tombol +/- agar tidak hitam */
[data-testid="stNumberInput"] button {
    background-color: #F1F5F9 !important;
    color: var(--text-main) !important;
    border-left: 1px solid var(--border) !important;
}
[data-testid="stNumberInput"] button:hover { background-color: #E2E8F0 !important; color: var(--primary) !important; }

/* Radio Button Styling (Fix Teks Hilang) */
[data-testid="stRadio"] > label { font-weight: 500 !important; font-size: 0.85rem !important; color: var(--text-muted) !important; }
[data-testid="stRadio"] div[role="radiogroup"] { gap: 12px !important; flex-direction: row !important; }
[data-testid="stRadio"] label {
    background: #F8FAFC !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 8px 24px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
}
/* Sembunyikan lingkaran radio */
[data-testid="stRadio"] label > div:first-child { display: none !important; }
/* Paksa teks gelap saat tidak dipilih */
[data-testid="stRadio"] label p { color: var(--text-main) !important; font-weight: 500 !important; margin: 0 !important; }
/* Saat dipilih */
[data-testid="stRadio"] label:has(input:checked) { background: var(--primary) !important; border-color: var(--primary) !important; }
[data-testid="stRadio"] label:has(input:checked) p { color: #FFFFFF !important; }

/* Primary Button */
.stButton > button {
    background: var(--primary) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 10px 24px !important;
    font-weight: 500 !important;
    font-size: 1rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2) !important;
}
.stButton > button:hover { background: var(--primary-hover) !important; transform: translateY(-1px); }

/* Metric Box */
[data-testid="stMetric"] {
    background: #F8FAFC !important;
    padding: 12px 16px !important;
    border-radius: 8px !important;
    border: 1px solid var(--border) !important;
}
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-weight: 500 !important; font-size: 0.85rem !important; }
[data-testid="stMetricValue"] { color: var(--text-main) !important; font-weight: 700 !important; font-size: 1.8rem !important; }

/* Result Card */
.result-card {
    background: #FFFFFF;
    border: 2px solid var(--primary);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: 1.5rem 0;
    box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.1);
}
.result-title { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); font-weight: 600; margin-bottom: 0.25rem; }
.result-value { font-size: 3rem; font-weight: 800; line-height: 1; color: var(--primary); }
.result-sub { font-size: 0.85rem; color: var(--text-muted); margin-top: 0.5rem; }

/* Progress Bar Customization */
.stProgress > div > div > div > div { background-color: var(--primary) !important; }
.stProgress > div > div > div { background-color: #E2E8F0 !important; }

/* Custom Table */
.custom-table { width: 100%; border-collapse: collapse; margin-top: 1rem; background: #FFFFFF; border-radius: 8px; overflow: hidden; border: 1px solid var(--border); }
.custom-table th { background: #F8FAFC; padding: 12px 16px; text-align: left; font-size: 0.85rem; color: var(--text-muted); font-weight: 600; border-bottom: 1px solid var(--border); }
.custom-table td { padding: 12px 16px; border-bottom: 1px solid var(--border); font-size: 0.95rem; color: var(--text-main); }
.custom-table tr:last-child td { border-bottom: none; }
.custom-table tr.pred-row td { background: #EFF6FF; font-weight: 600; color: var(--primary); }

/* Expander & Alert */
[data-testid="stExpander"] { border: 1px solid var(--border) !important; border-radius: 8px !important; background: #FFFFFF !important; }
[data-testid="stExpander"] summary { font-weight: 500 !important; color: var(--text-main) !important; }
[data-testid="stAlert"] { border-radius: 8px !important; background: #F8FAFC !important; border: 1px solid var(--border) !important; }
[data-testid="stAlert"] p { color: var(--text-main) !important; font-size: 0.9rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Load Model ───────────────────────────────────────────────────────────────
try:
    artefak = load_model()
except FileNotFoundError:
    st.error("Model tidak ditemukan. Jalankan `python train_model.py` dulu.")
    st.stop()

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>FitSize</h1>
    <p>Prediksi ukuran pakaian berdasarkan ukuran tubuh & standar EN 13402</p>
</div>
""", unsafe_allow_html=True)

# ── Form Input (Dibungkus Card) ──────────────────────────────────────────────
st.markdown('<div class="form-card">', unsafe_allow_html=True)
st.subheader("Data Tubuh")
st.markdown("<br>", unsafe_allow_html=True)

gender = st.radio("Jenis Kelamin", ["Pria", "Wanita"], horizontal=True)

col1, col2 = st.columns(2, gap="large")
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

st.markdown('</div>', unsafe_allow_html=True)

# ── Prediksi ─────────────────────────────────────────────────────────────────
if st.button("Prediksi Ukuran", type="primary", use_container_width=True):
    gender_int = 0 if gender == "Pria" else 1
    size, conf, bmi_val, dist = do_predict(
        artefak, gender_int, height, weight, chest, waist, hip)

    # Hasil Utama (Custom Card)
    st.markdown(f"""
    <div class="result-card">
        <div>
            <div class="result-title">Ukuran Rekomendasi</div>
            <div class="result-value">{size}</div>
            <div class="result-sub">Standar EN 13402 · {gender}</div>
        </div>
        <div style="text-align: right;">
            <div class="result-title">Tingkat Keyakinan</div>
            <div style="font-size: 2rem; font-weight: 700; color: var(--text-main); margin-top: 0.25rem;">{conf:.1f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Distribusi
    st.subheader("Distribusi Prediksi")
    st.caption("Berdasarkan 15 tetangga terdekat (k=15)")
    
    for s in SIZE_ORDER:
        pct = dist.get(s, 0)
        is_pred = (s == size)
        label = f"**{s}** (Prediksi)" if is_pred else s
        st.progress(int(pct), text=f"{label} — {pct:.1f}%")

    st.divider()

    # Tabel EN 13402 (Custom HTML)
    st.subheader(f"Rentang EN 13402 — {gender}")
    tbl = EN13402[gender]
    col_header = "Lingkar Dada" if gender == "Pria" else "Bust"
    
    table_html = f"<table class='custom-table'><thead><tr><th>Ukuran</th><th>{col_header}</th><th>Status</th></tr></thead><tbody>"
    for s in SIZE_ORDER:
        is_pred = (s == size)
        row_class = "pred-row" if is_pred else ""
        mark = "Rekomendasi" if is_pred else "—"
        table_html += f"<tr class='{row_class}'><td><strong>{s}</strong></td><td>{tbl[s]}</td><td>{mark}</td></tr>"
    table_html += "</tbody></table>"
    
    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        "Prediksi berdasarkan data CAESAR (Amerika/Eropa) & standar EN 13402. "
        "Ukuran aktual tiap merek pakaian bisa berbeda — gunakan sebagai referensi."
    )

# ── Info Model ───────────────────────────────────────────────────────────────
with st.expander("Tentang Model"):
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
    "Prediksi Ukuran Pakaian Menggunakan Algoritma K-Nearest Neighbor "
    "dengan Pembobotan Fitur Lingkar Dada pada Dataset CAESAR."
)
