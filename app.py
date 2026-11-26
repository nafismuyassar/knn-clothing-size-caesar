"""
app.py — Streamlit App: Prediksi Ukuran Pakaian (KNN + CAESAR)
Nafis Muyassar
"""
import pickle
from pathlib import Path
import numpy as np
import streamlit as st

# Menggunakan pathlib agar aman saat di-deploy di Streamlit Cloud
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


# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FitSize — Prediksi Ukuran Pakaian",
    page_icon="👕",
    layout="centered",
)


# ── CSS injection (VERSI PERBAIKAN) ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --primary: #4F46E5; /* Indigo solid, tidak neko-neko */
    --bg: #F8FAFC;
    --card: #FFFFFF;
    --text-main: #0F172A;
    --text-muted: #475569; /* Lebih gelap agar terbaca */
    --border: #E2E8F0;
}

/* ========== GLOBAL ========== */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-main);
}

.stApp {
    background-color: var(--bg) !important;
}

/* Sembunyikan elemen bawaan Streamlit */
#MainMenu, footer, header, [data-testid="stHeader"] {
    visibility: hidden !important;
}

.block-container {
    max-width: 760px !important;
    padding-top: 3rem !important;
}

/* ========== TYPOGRAPHY ========== */
h1, h2, h3 {
    font-weight: 700 !important;
    color: var(--text-main) !important;
    letter-spacing: -0.02em !important;
}
p {
    color: var(--text-main) !important;
}

/* ========== HERO ========== */
.hero {
    text-align: center;
    padding-bottom: 2rem;
}
.hero-icon {
    font-size: 3rem;
    display: inline-block;
    margin-bottom: 1rem;
}
.hero h1 {
    font-size: 2.5rem !important;
    color: var(--primary) !important;
    margin-bottom: 0.5rem !important;
}
.hero p {
    color: var(--text-muted);
    font-size: 1.05rem;
    max-width: 450px;
    margin: 0 auto;
}

/* ========== SECTION TITLE ========== */
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-main);
    margin: 1.5rem 0 1rem;
    display: flex;
    align-items: center;
}
.section-title::before {
    content: '';
    display: inline-block;
    width: 4px;
    height: 18px;
    background: var(--primary);
    border-radius: 2px;
    margin-right: 10px;
}

/* ========== INPUTS ========== */
[data-testid="stNumberInput"] label, [data-testid="stRadio"] > label {
    font-weight: 600 !important;
    color: var(--text-main) !important;
    font-size: 0.9rem !important;
    margin-bottom: 0.25rem !important;
}

[data-testid="stNumberInput"] input {
    border-radius: 8px !important;
    border: 1px solid var(--border) !important;
    padding: 0.6rem 0.8rem !important;
    background: var(--card) !important;
    color: var(--text-main) !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.15) !important;
}

/* Perbaikan tombol +/- agar tidak hitam */
[data-testid="stNumberInput"] button {
    background-color: #F1F5F9 !important;
    color: var(--text-main) !important;
    border-left: 1px solid var(--border) !important;
}
[data-testid="stNumberInput"] button:hover {
    background-color: #E2E8F0 !important;
    color: var(--primary) !important;
}
[data-testid="stNumberInput"] button svg {
    fill: var(--text-main) !important;
}

/* ========== RADIO BUTTON (PERBAIKAN TOTAL) ========== */
[data-testid="stRadio"] > div[role="radiogroup"] {
    gap: 1rem !important;
    flex-direction: row !important;
    flex-wrap: wrap !important;
}
[data-testid="stRadio"] label {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}
[data-testid="stRadio"] label:hover {
    border-color: var(--primary) !important;
    background: #EEF2FF !important;
}
/* Menyembunyikan lingkaran radio bawaan tapi teks tetap ada */
[data-testid="stRadio"] label > div:first-child {
    display: none !important;
}
[data-testid="stRadio"] label div[data-testid="stMarkdownContainer"] p {
    color: var(--text-muted) !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    margin: 0 !important;
}
/* Saat terpilih */
[data-testid="stRadio"] label:has(input:checked) {
    border-color: var(--primary) !important;
    background: var(--primary) !important;
}
[data-testid="stRadio"] label:has(input:checked) div[data-testid="stMarkdownContainer"] p {
    color: white !important;
}

/* ========== BUTTON ========== */
.stButton > button {
    background: var(--primary) !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    color: white !important;
    box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.2) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #4338CA !important;
    box-shadow: 0 6px 8px -1px rgba(79, 70, 229, 0.3) !important;
    transform: translateY(-1px);
}

/* ========== METRIC (BMI) ========== */
[data-testid="stMetric"] {
    background: var(--card) !important;
    padding: 16px !important;
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    color: var(--primary) !important;
    font-weight: 700 !important;
}

/* ========== RESULT CARD ========== */
.result-card {
    background: var(--card);
    border: 2px solid var(--primary);
    border-radius: 16px;
    padding: 2rem;
    color: var(--text-main);
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
    margin: 1.5rem 0;
}
.result-label {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.result-size {
    font-size: 3.5rem;
    font-weight: 800;
    line-height: 1;
    color: var(--primary);
}
.result-standard {
    font-size: 0.85rem;
    color: var(--text-muted);
    margin-top: 0.5rem;
}
.result-right { text-align: right; }
.conf-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-main);
    margin: 0.25rem 0;
}
.conf-bar {
    width: 120px;
    height: 8px;
    background: #E2E8F0;
    border-radius: 4px;
    overflow: hidden;
    margin-left: auto;
}
.conf-fill {
    height: 100%;
    background: var(--primary);
    border-radius: 4px;
}

/* ========== DISTRIBUTION BARS ========== */
.dist-list {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
}
.dist-row {
    display: flex;
    align-items: center;
    gap: 15px;
    margin: 12px 0;
}
.dist-label { width: 40px; font-weight: 600; font-size: 0.9rem; }
.dist-bar { flex: 1; height: 8px; background: #F1F5F9; border-radius: 4px; overflow: hidden; }
.dist-fill { height: 100%; border-radius: 4px; background: #CBD5E1; }
.dist-row.pred .dist-fill { background: var(--primary); }
.dist-pct { width: 50px; text-align: right; font-weight: 600; font-size: 0.85rem; color: var(--text-muted); }
.dist-row.pred .dist-pct, .dist-row.pred .dist-label { color: var(--primary); }

/* ========== TABLE ========== */
.size-table-wrap {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}
.size-table-wrap table { width: 100%; border-collapse: collapse; }
.size-table-wrap th {
    background: #F8FAFC; padding: 12px 16px; text-align: left;
    font-size: 0.8rem; color: var(--text-muted); border-bottom: 1px solid var(--border);
}
.size-table-wrap td {
    padding: 12px 16px; border-bottom: 1px solid var(--border); font-size: 0.95rem;
}
.size-table-wrap tr:last-child td { border-bottom: none; }
.size-table-wrap tr.pred-row td { background: #EEF2FF; font-weight: 600; color: var(--primary); }

/* ========== EXPANDER ========== */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    background: var(--card) !important;
}
[data-testid="stExpander"] summary {
    padding: 12px 16px !important;
    font-weight: 600 !important;
}

/* ========== INFO ========== */
div[data-testid="stAlert"] {
    border-radius: 8px !important;
    border: 1px solid #BFDBFE !important;
    background: #EFF6FF !important;
}
div[data-testid="stAlert"] p { color: #1E40AF !important; font-size: 0.9rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Load model ──────────────────────────────────────────────────────────────
try:
    artefak = load_model()
except FileNotFoundError:
    st.error("Model tidak ditemukan. Pastikan folder `model/model.pkl` sudah di-commit ke repo GitHub.")
    st.stop()


# ── Hero header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-icon">👕</div>
    <h1>FitSize</h1>
    <p>Prediksi ukuran pakaian yang pas untukmu — berdasarkan ukuran tubuh & standar EN 13402</p>
</div>
""", unsafe_allow_html=True)

st.divider()


# ── Form input ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Data Tubuh</div>', unsafe_allow_html=True)

# Perbaikan: Streamlit punya st.segmented_control (versi baru) yang jauh lebih bagus dari radio
# Jika versi streamlit Anda di Cloud mendukung, ini akan otomatis terpakai.
try:
    gender = st.segmented_control("Jenis Kelamin", ["Pria", "Wanita"], default="Pria", label_visibility="collapsed")
    if not gender: gender = "Pria" # Fallback jika user nge-klik sampai kosong
except AttributeError:
    gender = st.radio("Jenis Kelamin", ["Pria", "Wanita"], horizontal=True, label_visibility="collapsed")

col1, col2 = st.columns(2, gap="medium")
with col1:
    height = st.number_input("Tinggi Badan (cm)",       min_value=100.0, max_value=250.0, value=170.0, step=0.5)
    weight = st.number_input("Berat Badan (kg)",         min_value=30.0,  max_value=200.0, value=50.0,  step=0.5)
    chest  = st.number_input("Lingkar Dada / Bust (cm)", min_value=50.0,  max_value=200.0, value=96.0,  step=0.5)
with col2:
    waist  = st.number_input("Lingkar Pinggang (cm)",    min_value=40.0,  max_value=200.0, value=90.0,  step=0.5)
    hip    = st.number_input("Lingkar Pinggul (cm)",     min_value=50.0,  max_value=200.0, value=100.0, step=0.5)
    bmi_now = weight / (height / 100) ** 2
    st.metric("BMI (otomatis)", f"{bmi_now:.1f}", help=bmi_category(bmi_now), delta_color="off")

st.divider()


# ── Prediksi ────────────────────────────────────────────────────────────────
if st.button("Prediksi Ukuran Saya", type="primary", use_container_width=True):
    gender_int = 0 if gender == "Pria" else 1
    size, conf, bmi_val, dist = do_predict(
        artefak, gender_int, height, weight, chest, waist, hip)

    st.markdown(f"""
    <div class="result-card">
        <div class="result-left">
            <div class="result-label">Ukuran Rekomendasi</div>
            <div class="result-size">{size}</div>
            <div class="result-standard">Standar EN 13402 · {gender}</div>
        </div>
        <div class="result-right">
            <div class="result-label">Kepercayaan</div>
            <div class="conf-value">{conf:.1f}%</div>
            <div class="conf-bar"><div class="conf-fill" style="width:{conf:.1f}%"></div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Distribusi Prediksi</div>', unsafe_allow_html=True)

    rows = ""
    for s in SIZE_ORDER:
        pct = dist.get(s, 0)
        is_pred = (s == size)
        row_class = "dist-row pred" if is_pred else "dist-row"
        rows += f"""
        <div class="{row_class}">
            <div class="dist-label">{s}</div>
            <div class="dist-bar"><div class="dist-fill" style="width:{pct:.1f}%;"></div></div>
            <div class="dist-pct">{pct:.1f}%</div>
        </div>
        """

    st.markdown(f"""
    <div class="dist-list">
        {rows}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f'<div class="section-title">Rentang EN 13402 — {gender}</div>', unsafe_allow_html=True)

    tbl = EN13402[gender]
    col_header = "Lingkar Dada" if gender == "Pria" else "Bust"

    table_rows = ""
    for s in SIZE_ORDER:
        is_pred = (s == size)
        row_class = 'pred-row' if is_pred else ''
        mark = "✅ Rekomendasi" if is_pred else "—"
        table_rows += f"""
        <tr class="{row_class}">
            <td><strong>{s}</strong></td>
            <td>{tbl[s]}</td>
            <td>{mark}</td>
        </tr>
        """

    st.markdown(f"""
    <div class="size-table-wrap">
        <table>
            <thead>
                <tr>
                    <th>Ukuran</th>
                    <th>{col_header}</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("Catatan: Prediksi berdasarkan data CAESAR (Amerika/Eropa) & standar EN 13402. Ukuran aktual tiap merek pakaian bisa berbeda.")


# ── Info model ──────────────────────────────────────────────────────────────
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
