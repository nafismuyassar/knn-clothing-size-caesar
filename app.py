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


# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FitSize — Prediksi Ukuran Pakaian",
    page_icon="👕",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ── CSS injection ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --primary: #4F46E5;
    --primary-2: #7C3AED;
    --bg: #F8FAFC;
    --card: #FFFFFF;
    --text: #0F172A;
    --muted: #64748B;
    --border: #E2E8F0;
}

/* ========== GLOBAL ========== */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text);
}

.stApp {
    background:
        radial-gradient(circle at 15% 0%, rgba(79, 70, 229, 0.10) 0%, transparent 45%),
        radial-gradient(circle at 85% 100%, rgba(124, 58, 237, 0.10) 0%, transparent 45%),
        #F8FAFC !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header, [data-testid="stHeader"] {
    visibility: hidden !important;
    height: 0 !important;
}

/* Container width */
.block-container, [data-testid="stMainBlockContainer"] {
    max-width: 840px !important;
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
}

/* ========== TYPOGRAPHY ========== */
h1, h2, h3 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    color: var(--text) !important;
    letter-spacing: -0.02em !important;
}

.stCaption, [data-testid="stCaptionContainer"] {
    color: var(--muted) !important;
    font-size: 0.85rem !important;
    line-height: 1.5 !important;
}

/* ========== HERO ========== */
.hero {
    text-align: center;
    padding: 1.5rem 1rem 1rem;
}
.hero-icon {
    font-size: 3.5rem;
    display: inline-block;
    animation: float 3s ease-in-out infinite;
}
@keyframes float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-6px); }
}
.hero h1 {
    font-size: 3rem !important;
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 55%, #EC4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0.25rem 0 0.75rem !important;
    font-weight: 800 !important;
}
.hero p {
    color: var(--muted);
    font-size: 1rem;
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ========== DIVIDER ========== */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, #E2E8F0 20%, #E2E8F0 80%, transparent) !important;
    margin: 1.75rem 0 !important;
}

/* ========== SECTION TITLE ========== */
.section-title {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text);
    margin: 1.5rem 0 1rem;
}
.section-title::before {
    content: '';
    width: 4px;
    height: 20px;
    background: linear-gradient(180deg, var(--primary), var(--primary-2));
    border-radius: 4px;
}

/* ========== INPUTS ========== */
[data-testid="stNumberInput"] label,
[data-testid="stRadio"] label p {
    font-weight: 600 !important;
    color: var(--text) !important;
    font-size: 0.875rem !important;
}

[data-testid="stNumberInput"] input {
    border-radius: 10px !important;
    border: 1.5px solid var(--border) !important;
    padding: 0.55rem 0.85rem !important;
    font-weight: 500 !important;
    background: var(--card) !important;
    transition: all 0.2s ease !important;
    font-size: 0.95rem !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.12) !important;
}
[data-testid="stNumberInput"] button {
    background: transparent !important;
    border: none !important;
    color: var(--muted) !important;
    border-radius: 6px !important;
}
[data-testid="stNumberInput"] button:hover {
    background: rgba(79, 70, 229, 0.08) !important;
    color: var(--primary) !important;
}

/* Radio as pills */
[data-testid="stRadio"] > div {
    gap: 8px !important;
    flex-wrap: wrap;
}
[data-testid="stRadio"] label {
    background: var(--card) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 999px !important;
    padding: 8px 20px !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
    margin-right: 0 !important;
}
[data-testid="stRadio"] label:hover {
    border-color: var(--primary) !important;
    background: #EEF2FF !important;
}
[data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(135deg, var(--primary), var(--primary-2)) !important;
    border-color: transparent !important;
}
[data-testid="stRadio"] label:has(input:checked) p {
    color: white !important;
}
[data-testid="stRadio"] label > div:first-of-type {
    display: none !important;
}

/* ========== BUTTON ========== */
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 24px !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    color: white !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35) !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.01em !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="baseButton-primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(79, 70, 229, 0.45) !important;
}
.stButton > button[kind="primary"]:active,
.stButton > button[data-testid="baseButton-primary"]:active {
    transform: translateY(0);
}

/* ========== METRIC (BMI) ========== */
[data-testid="stMetric"] {
    background: var(--card) !important;
    padding: 16px 20px !important;
    border-radius: 14px !important;
    border: 1px solid var(--border) !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
}
[data-testid="stMetricValue"] {
    color: var(--primary) !important;
    font-weight: 700 !important;
    font-size: 1.6rem !important;
}

/* ========== RESULT CARD ========== */
.result-card {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
    border-radius: 20px;
    padding: 1.75rem 2rem;
    color: white;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 2rem;
    box-shadow: 0 10px 40px -10px rgba(79, 70, 229, 0.45);
    margin: 1rem 0 1.5rem;
    position: relative;
    overflow: hidden;
    animation: slideUp 0.4s ease;
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}
.result-card::before {
    content: '';
    position: absolute;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(255,255,255,0.18), transparent 70%);
    top: -140px; right: -80px;
    border-radius: 50%;
    pointer-events: none;
}
.result-card::after {
    content: '';
    position: absolute;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(236,72,153,0.25), transparent 70%);
    bottom: -120px; left: -60px;
    border-radius: 50%;
    pointer-events: none;
}
.result-left, .result-right { position: relative; z-index: 1; }
.result-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    opacity: 0.85;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.result-size {
    font-size: 3.5rem;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -0.03em;
    color: white;
}
.result-standard {
    font-size: 0.8rem;
    opacity: 0.75;
    margin-top: 0.5rem;
    font-weight: 500;
}
.result-right { text-align: right; }
.conf-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    opacity: 0.85;
    font-weight: 700;
}
.conf-value {
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0.15rem 0 0.5rem;
}
.conf-bar {
    width: 130px;
    height: 6px;
    background: rgba(255,255,255,0.25);
    border-radius: 999px;
    overflow: hidden;
    margin-left: auto;
}
.conf-fill {
    height: 100%;
    background: white;
    border-radius: 999px;
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}

@media (max-width: 600px) {
    .result-card { flex-direction: column; text-align: center; gap: 1rem; }
    .result-right { text-align: center; }
    .conf-bar { margin: 0 auto; }
    .hero h1 { font-size: 2.2rem !important; }
    .result-size { font-size: 2.8rem; }
}

/* ========== DISTRIBUTION BARS ========== */
.dist-list {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.25rem 1.25rem 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    margin-top: 0.5rem;
}
.dist-header {
    display: flex;
    justify-content: space-between;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    font-weight: 700;
    margin-bottom: 0.75rem;
}
.dist-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 10px 0;
}
.dist-label {
    width: 50px;
    font-weight: 700;
    font-size: 0.9rem;
    color: var(--text);
    white-space: nowrap;
}
.dist-bar {
    flex: 1;
    height: 10px;
    background: #F1F5F9;
    border-radius: 999px;
    overflow: hidden;
}
.dist-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.7s cubic-bezier(0.4, 0, 0.2, 1);
}
.dist-pct {
    width: 55px;
    text-align: right;
    font-weight: 600;
    font-size: 0.8rem;
    color: var(--muted);
}
.dist-row.pred .dist-label,
.dist-row.pred .dist-pct {
    color: var(--primary);
}

/* ========== SIZE TABLE ========== */
.size-table-wrap {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    margin-top: 0.5rem;
}
.size-table-wrap table {
    width: 100%;
    border-collapse: collapse;
    margin: 0 !important;
}
.size-table-wrap thead th {
    background: #F8FAFC !important;
    color: var(--muted) !important;
    text-align: left;
    padding: 12px 18px !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
    border: none !important;
    border-bottom: 1px solid var(--border) !important;
}
.size-table-wrap tbody td {
    padding: 13px 18px !important;
    border: none !important;
    border-bottom: 1px solid #F1F5F9 !important;
    font-size: 0.92rem;
    color: var(--text);
}
.size-table-wrap tbody tr:last-child td {
    border-bottom: none !important;
}
.size-table-wrap tbody tr.pred-row td {
    background: linear-gradient(90deg, rgba(79, 70, 229, 0.08), rgba(124, 58, 237, 0.08)) !important;
    font-weight: 700 !important;
    color: var(--primary) !important;
}

/* ========== INFO / ALERT ========== */
div[data-testid="stAlert"] {
    border-radius: 14px !important;
    border: 1px solid #BFDBFE !important;
    background: #EFF6FF !important;
    padding: 14px 18px !important;
}
div[data-testid="stAlert"] p {
    color: #1E40AF !important;
    font-size: 0.875rem !important;
    line-height: 1.5 !important;
    font-weight: 500 !important;
}

/* ========== EXPANDER ========== */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    background: var(--card) !important;
    overflow: hidden;
    margin-top: 1.5rem;
}
[data-testid="stExpander"] summary {
    padding: 14px 20px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}
[data-testid="stExpander"] summary:hover {
    background: #F8FAFC !important;
}

/* Markdown table inside expander */
[data-testid="stExpander"] .stMarkdown table {
    border-collapse: separate !important;
    border-spacing: 0 !important;
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid var(--border) !important;
    width: 100%;
    margin-top: 0.5rem;
}
[data-testid="stExpander"] .stMarkdown thead th {
    background: #F8FAFC !important;
    padding: 10px 14px !important;
    text-align: left !important;
    font-weight: 700 !important;
    color: var(--muted) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    border: none !important;
    border-bottom: 1px solid var(--border) !important;
}
[data-testid="stExpander"] .stMarkdown tbody td {
    padding: 10px 14px !important;
    border: none !important;
    border-bottom: 1px solid #F1F5F9 !important;
    font-size: 0.88rem;
}
[data-testid="stExpander"] .stMarkdown tbody tr:last-child td {
    border-bottom: none !important;
}

/* ========== FOOTER ========== */
.app-footer {
    text-align: center;
    color: var(--muted);
    font-size: 0.8rem;
    line-height: 1.6;
    padding: 1.5rem 1rem 0;
}
</style>
""", unsafe_allow_html=True)


# ── Load model ──────────────────────────────────────────────────────────────
try:
    artefak = load_model()
except FileNotFoundError:
    st.error("Model tidak ditemukan. Jalankan `python train_model.py` dulu.")
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
st.markdown('<div class="section-title">📋 Data Tubuh</div>', unsafe_allow_html=True)

gender = st.radio("Jenis Kelamin", ["Pria", "Wanita"], horizontal=True)

col1, col2 = st.columns(2, gap="medium")
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


# ── Prediksi ────────────────────────────────────────────────────────────────
if st.button("🔍  Prediksi Ukuran Saya", type="primary", use_container_width=True):
    gender_int = 0 if gender == "Pria" else 1
    size, conf, bmi_val, dist = do_predict(
        artefak, gender_int, height, weight, chest, waist, hip)

    # Result card
    st.markdown(f"""
    <div class="result-card">
        <div class="result-left">
            <div class="result-label">🎯 Ukuran Rekomendasi</div>
            <div class="result-size">{size}</div>
            <div class="result-standard">Standar EN 13402 · {gender}</div>
        </div>
        <div class="result-right">
            <div class="conf-label">Kepercayaan</div>
            <div class="conf-value">{conf:.1f}%</div>
            <div class="conf-bar"><div class="conf-fill" style="width:{conf:.1f}%"></div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Distribution
    st.markdown('<div class="section-title">📊 Distribusi Prediksi</div>', unsafe_allow_html=True)

    rows = ""
    for s in SIZE_ORDER:
        pct = dist.get(s, 0)
        is_pred = (s == size)
        row_class = "dist-row pred" if is_pred else "dist-row"
        fill_color = ("linear-gradient(90deg, #4F46E5, #7C3AED)"
                      if is_pred else "linear-gradient(90deg, #CBD5E1, #94A3B8)")
        rows += f"""
        <div class="{row_class}">
            <div class="dist-label">{s}</div>
            <div class="dist-bar"><div class="dist-fill" style="width:{pct:.1f}%; background:{fill_color};"></div></div>
            <div class="dist-pct">{pct:.1f}%</div>
        </div>
        """

    st.markdown(f"""
    <div class="dist-list">
        <div class="dist-header"><span>Ukuran</span><span>Probabilitas</span></div>
        {rows}
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Table EN 13402
    st.markdown(f'<div class="section-title">📏 Rentang EN 13402 — {gender}</div>', unsafe_allow_html=True)

    tbl = EN13402[gender]
    col_header = "Lingkar Dada" if gender == "Pria" else "Bust"

    table_rows = ""
    for s in SIZE_ORDER:
        is_pred = (s == size)
        row_class = 'pred-row' if is_pred else ''
        mark = "✅ Kamu" if is_pred else "—"
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
                    <th></th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.info(
        "💡 **Catatan:** Prediksi berdasarkan data CAESAR (Amerika/Eropa) & standar EN 13402. "
        "Ukuran aktual tiap merek pakaian bisa berbeda — gunakan sebagai referensi."
    )


# ── Info model ──────────────────────────────────────────────────────────────
with st.expander("ℹ️  Tentang Model"):
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


# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    <em>Prediksi Ukuran Pakaian Menggunakan Algoritma K-Nearest Neighbor<br>
    dengan Pembobotan Fitur Lingkar Dada pada Dataset CAESAR</em>
</div>
""", unsafe_allow_html=True)
