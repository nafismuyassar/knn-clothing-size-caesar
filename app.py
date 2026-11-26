"""
app.py — Streamlit App: Prediksi Ukuran Pakaian (KNN + CAESAR)
Nafis Muyassar
"""
import pickle
from pathlib import Path
import numpy as np
import streamlit as st

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
    page_icon="📏", # Menggunakan ikon penggaris (standar, bukan emoji ekspresif)
    layout="centered",
)

# Memaksa Streamlit menggunakan Light Mode (mencegah bug hitam di form)
st.markdown("""
<script>
    // Script kecil untuk memastikan tema terang
    if (window.parent.document.documentElement.getAttribute('data-theme') === 'dark') {
        window.parent.document.documentElement.setAttribute('data-theme', 'light');
    }
</script>
""", unsafe_allow_html=True)

# ── CSS injection ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

:root {
    --primary: #4F46E5;
    --primary-hover: #4338CA;
    --bg: #F4F7FB; /* Background abu kebiruan yang sangat halus */
    --card-bg: #FFFFFF;
    --text-main: #1E293B; /* Slate gelap untuk teks utama */
    --text-muted: #64748B;
    --border: #E2E8F0;
}

/* ========== GLOBAL & TYPOGRAPHY ========== */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
    color: var(--text-main);
}
.stApp { background-color: var(--bg) !important; }

/* Sembunyikan elemen Streamlit */
#MainMenu, footer, header, [data-testid="stHeader"] { visibility: hidden !important; }

.block-container {
    max-width: 720px !important;
    padding-top: 3rem !important;
    padding-bottom: 2rem !important;
}

h1, h2, h3 { font-weight: 600 !important; color: var(--text-main) !important; letter-spacing: -0.02em !important; }

/* ========== HERO ========== */
.hero { text-align: center; margin-bottom: 2rem; }
.hero h1 {
    font-size: 2.8rem !important;
    color: var(--text-main) !important;
    margin-bottom: 0.5rem !important;
    font-weight: 700 !important;
}
.hero p { color: var(--text-muted); font-size: 1.05rem; max-width: 450px; margin: 0 auto; line-height: 1.6; }

/* ========== SECTION TITLE ========== */
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-main);
    margin: 0 0 1.5rem 0;
    display: flex;
    align-items: center;
}
.section-title::before {
    content: ''; display: inline-block; width: 4px; height: 18px;
    background: var(--primary); border-radius: 2px; margin-right: 10px;
}

/* ========== FORM CARD ========== */
.form-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.01);
    margin-bottom: 2rem;
}

/* ========== INPUTS (Memperbaiki bug hitam) ========== */
[data-testid="stNumberInput"] label, [data-testid="stRadio"] > label {
    font-weight: 500 !important;
    color: var(--text-main) !important;
    font-size: 0.9rem !important;
    margin-bottom: 0.4rem !important;
}

/* Input Field */
[data-testid="stNumberInput"] input {
    border-radius: 8px !important;
    border: 1px solid var(--border) !important;
    padding: 0.6rem 0.8rem !important;
    background-color: #F8FAFC !important;
    color: var(--text-main) !important;
    font-weight: 500 !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: var(--primary) !important;
    background-color: #FFFFFF !important;
    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1) !important;
}

/* Tombol +/- (FIX BUG HITAM) */
[data-testid="stNumberInput"] button {
    background-color: #F1F5F9 !important;
    color: var(--text-main) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}
[data-testid="stNumberInput"] button:hover {
    background-color: #E2E8F0 !important;
    color: var(--primary) !important;
}

/* ========== RADIO BUTTON (FIX BUG TEKS HILANG) ========== */
[data-testid="stRadio"] > div[role="radiogroup"] {
    gap: 10px !important;
    flex-direction: row !important;
}
[data-testid="stRadio"] label {
    background: #F8FAFC !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}
[data-testid="stRadio"] label:hover { border-color: var(--primary) !important; }

/* Sembunyikan lingkaran radio */
[data-testid="stRadio"] label > div:first-child { display: none !important; }

/* Paksa teks selalu berwarna gelap saat TIDAK terpilih */
[data-testid="stRadio"] label [data-testid="stMarkdownContainer"] p {
    color: var(--text-main) !important;
    font-weight: 500 !important;
    margin: 0 !important;
}

/* Saat TERPILIH */
[data-testid="stRadio"] label:has(input:checked) {
    background: var(--primary) !important;
    border-color: var(--primary) !important;
}
[data-testid="stRadio"] label:has(input:checked) [data-testid="stMarkdownContainer"] p {
    color: #FFFFFF !important;
    font-weight: 600 !important;
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
    width: 100%;
}
.stButton > button:hover {
    background: var(--primary-hover) !important;
    transform: translateY(-1px);
    box-shadow: 0 6px 8px -1px rgba(79, 70, 229, 0.3) !important;
}

/* ========== METRIC (BMI) ========== */
[data-testid="stMetric"] {
    background: #F8FAFC !important;
    padding: 16px !important;
    border-radius: 8px !important;
    border: 1px solid var(--border) !important;
}
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-weight: 500 !important; }
[data-testid="stMetricValue"] { color: var(--text-main) !important; font-weight: 700 !important; }

/* ========== HASIL PREDIKSI ========== */
.result-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05);
    margin-bottom: 2rem;
}
.result-label {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.result-size { font-size: 3.5rem; font-weight: 700; line-height: 1; color: var(--primary); }
.result-standard { font-size: 0.85rem; color: var(--text-muted); margin-top: 0.5rem; }
.result-right { text-align: right; }
.conf-value { font-size: 2rem; font-weight: 600; color: var(--text-main); margin: 0.25rem 0; }
.conf-bar { width: 120px; height: 6px; background: #E2E8F0; border-radius: 3px; overflow: hidden; margin-left: auto; }
.conf-fill { height: 100%; background: var(--primary); border-radius: 3px; }

/* ========== DISTRIBUTION BARS ========== */
.dist-list { background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; margin-bottom: 2rem; }
.dist-row { display: flex; align-items: center; gap: 15px; margin: 12px 0; }
.dist-label { width: 40px; font-weight: 500; font-size: 0.9rem; color: var(--text-muted); }
.dist-bar { flex: 1; height: 6px; background: #F1F5F9; border-radius: 3px; overflow: hidden; }
.dist-fill { height: 100%; border-radius: 3px; background: #CBD5E1; }
.dist-row.pred .dist-fill { background: var(--primary); }
.dist-pct { width: 45px; text-align: right; font-weight: 500; font-size: 0.85rem; color: var(--text-muted); }
.dist-row.pred .dist-pct, .dist-row.pred .dist-label { color: var(--primary); font-weight: 600; }

/* ========== TABLE ========== */
.size-table-wrap { background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; margin-bottom: 1.5rem; }
.size-table-wrap table { width: 100%; border-collapse: collapse; }
.size-table-wrap th { background: #F8FAFC; padding: 14px 16px; text-align: left; font-size: 0.85rem; color: var(--text-muted); border-bottom: 1px solid var(--border); font-weight: 500; }
.size-table-wrap td { padding: 14px 16px; border-bottom: 1px solid var(--border); font-size: 0.95rem; color: var(--text-main); }
.size-table-wrap tr:last-child td { border-bottom: none; }
.size-table-wrap tr.pred-row td { background: #EEF2FF; font-weight: 600; color: var(--primary); }

/* ========== EXPANDER & INFO ========== */
[data-testid="stExpander"] { border: 1px solid var(--border) !important; border-radius: 12px !important; background: var(--card-bg) !important; }
[data-testid="stExpander"] summary { padding: 12px 16px !important; font-weight: 500 !important; color: var(--text-main) !important; }
div[data-testid="stAlert"] { border-radius: 8px !important; border: 1px solid #BFDBFE !important; background: #EFF6FF !important; }
div[data-testid="stAlert"] p { color: #1E40AF !important; font-size: 0.9rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Load model ──────────────────────────────────────────────────────────────
try:
    artefak = load_model()
except FileNotFoundError:
    st.error("Model tidak ditemukan. Pastikan folder `model/model.pkl` sudah di-commit ke repo GitHub Anda.")
    st.stop()


# ── Hero header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>FitSize</h1>
    <p>Prediksi ukuran pakaian yang pas untukmu — berdasarkan ukuran tubuh & standar EN 13402</p>
</div>
""", unsafe_allow_html=True)


# ── Form Input (Dibungkus dalam Card) ───────────────────────────────────────
st.markdown('<div class="form-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Data Tubuh</div>', unsafe_allow_html=True)

gender = st.radio("Jenis Kelamin", ["Pria", "Wanita"], horizontal=True)

col1, col2 = st.columns(2, gap="large")
with col1:
    height = st.number_input("Tinggi Badan (cm)",       min_value=100.0, max_value=250.0, value=170.0, step=0.5)
    weight = st.number_input("Berat Badan (kg)",         min_value=30.0,  max_value=200.0, value=50.0,  step=0.5)
    chest  = st.number_input("Lingkar Dada / Bust (cm)", min_value=50.0,  max_value=200.0, value=96.0,  step=0.5)
with col2:
    waist  = st.number_input("Lingkar Pinggang (cm)",    min_value=40.0,  max_value=200.0, value=90.0,  step=0.5)
    hip    = st.number_input("Lingkar Pinggul (cm)",     min_value=50.0,  max_value=200.0, value=100.0, step=0.5)
    
    # BMI Metric
    bmi_now = weight / (height / 100) ** 2
    st.metric("BMI (otomatis)", f"{bmi_now:.1f}", help=bmi_category(bmi_now), delta_color="off")
    
st.markdown('</div>', unsafe_allow_html=True)


# ── Prediksi ────────────────────────────────────────────────────────────────
if st.button("Prediksi Ukuran Saya", type="primary"):
    gender_int = 0 if gender == "Pria" else 1
    size, conf, bmi_val, dist = do_predict(
        artefak, gender_int, height, weight, chest, waist, hip)

    # Result Card
    st.markdown(f"""
    <div class="result-card">
        <div class="result-left">
            <div class="result-label">Rekomendasi Ukuran</div>
            <div class="result-size">{size}</div>
            <div class="result-standard">Standar EN 13402 · {gender}</div>
        </div>
        <div class="result-right">
            <div class="result-label">Tingkat Keyakinan</div>
            <div class="conf-value">{conf:.1f}%</div>
            <div class="conf-bar"><div class="conf-fill" style="width:{conf:.1f}%"></div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Distribution
    st.markdown('<div class="section-title">Distribusi Probabilitas</div>', unsafe_allow_html=True)
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
    st.markdown(f'<div class="dist-list">{rows}</div>', unsafe_allow_html=True)

    # Table EN 13402
    st.markdown(f'<div class="section-title">Rentang EN 13402 — {gender}</div>', unsafe_allow_html=True)
    tbl = EN13402[gender]
    col_header = "Lingkar Dada" if gender == "Pria" else "Bust"

    table_rows = ""
    for s in SIZE_ORDER:
        is_pred = (s == size)
        row_class = 'pred-row' if is_pred else ''
        mark = "Rekomendasi" if is_pred else "—"
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
                <tr><th>Ukuran</th><th>{col_header}</th><th>Status</th></tr>
            </thead>
            <tbody>{table_rows}</tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

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
