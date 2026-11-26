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


def html_table(headers, rows, hl=None):
    """Tabel HTML dengan warna eksplisit agar selalu terbaca."""
    th = "".join(f"<th>{h}</th>" for h in headers)
    body = ""
    for i, row in enumerate(rows):
        cls = ' class="hl"' if i == hl else ""
        body += f"<tr{cls}>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
    return f'<table class="tbl"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>'


def dist_bars(dist, size):
    out = ""
    for s in SIZE_ORDER:
        pct = max(0.0, min(100.0, dist.get(s, 0)))
        on = " on" if s == size else ""
        badge = '<span class="badge">Prediksi</span>' if s == size else ""
        out += (
            f'<div class="bar-row{on}"><div class="bar-name">{s}</div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%"></div></div>'
            f'<div class="bar-val">{pct:.1f}%</div>{badge}</div>'
        )
    return out


# ── Setup halaman ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Ukuran Pakaian",
    page_icon=None,
    layout="wide",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --ink: #111418;
    --muted: #4A5361;
    --line: #E3E6EB;
    --grad: linear-gradient(135deg, #0A0A0F 0%, #10246B 55%, #2563EB 100%);
    --grad-flat: linear-gradient(90deg, #14338A 0%, #2563EB 100%);
}

/* Latar & font */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background: #FFFFFF !important;
}
.stApp, .stApp p, .stApp label, .stApp li, .stApp button, .stApp input,
.hero, .hero *, .card, .card *, .res, .res *, .tbl, .tbl * {
    font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
}
.block-container { max-width: 1120px; padding-top: 2rem; padding-bottom: 3rem; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { visibility: hidden; }

/* Teks bawaan Streamlit: selalu gelap di atas putih */
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *,
[data-testid="stRadio"] label, [data-testid="stRadio"] label *,
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] strong {
    color: var(--ink) !important;
}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {
    color: var(--muted) !important;
}

/* Hero */
.hero {
    background: var(--grad);
    border-radius: 18px;
    padding: 2.2rem 2.4rem;
    margin-bottom: 1.5rem;
}
.hero-title { color: #FFFFFF; font-size: 2.1rem; font-weight: 700; letter-spacing: -0.02em; }
.hero-sub   { color: #D6DFF2; font-size: 1rem; margin-top: .4rem; }

/* Kartu */
.card {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1rem;
}
.sec { color: var(--ink); font-size: 1.1rem; font-weight: 700; margin-bottom: .2rem; }
.sec-sub { color: var(--muted); font-size: .88rem; margin-bottom: .9rem; }

/* Wadah input */
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid var(--line) !important;
    border-radius: 16px !important;
    background: #FFFFFF !important;
}

/* Pilihan jenis kelamin: lingkaran hitam, terpilih biru */
[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
    background-color: #FFFFFF !important;
    border: 2px solid #111418 !important;
    box-shadow: none !important;
}
[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) > div:first-child {
    background-color: #2563EB !important;
    border-color: #2563EB !important;
}

/* Kolom angka: satu garis tepi dengan ketebalan sama di semua sisi */
[data-testid="stNumberInput"] [data-baseweb="input"] {
    background: #FFFFFF !important;
    border: 1.5px solid #111418 !important;
    border-radius: 10px !important;
    box-shadow: none !important;
    outline: none !important;
    overflow: hidden;
    height: 2.6rem;
    align-items: stretch;
}
[data-testid="stNumberInput"] [data-baseweb="input"]:focus-within {
    border-color: #2563EB !important;
}
[data-testid="stNumberInput"] [data-baseweb="base-input"] {
    background: #FFFFFF !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}
[data-testid="stNumberInput"] input {
    background: #FFFFFF !important;
    color: #111418 !important;
    -webkit-text-fill-color: #111418 !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    font-weight: 500;
}

/* Tombol plus dan minus: kotak seragam, pas dengan tinggi kolom */
[data-testid="stNumberInput"] [data-baseweb="input"] button {
    width: 2.4rem !important;
    min-width: 2.4rem !important;
    height: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
    display: flex !important;
    align-items: center;
    justify-content: center;
    background: #F1F3F6 !important;
    color: #111418 !important;
    border: none !important;
    border-left: 1px solid #111418 !important;
    border-radius: 0 !important;
}
[data-testid="stNumberInput"] [data-baseweb="input"] button:hover {
    background: #E2E7EF !important;
}
[data-testid="stNumberInput"] [data-baseweb="input"] button svg {
    width: 14px !important;
    height: 14px !important;
    fill: #111418 !important;
    color: #111418 !important;
}

/* Tombol utama */
.stButton > button {
    background: var(--grad-flat) !important;
    border: none !important;
    border-radius: 12px !important;
    height: 3.1rem;
    box-shadow: 0 6px 16px rgba(37, 99, 235, .25);
}
.stButton > button, .stButton > button * { color: #FFFFFF !important; font-weight: 600; }
.stButton > button:hover { filter: brightness(1.08); }

/* Hasil utama */
.res {
    background: var(--grad);
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1rem;
    display: flex; justify-content: space-between; align-items: center; gap: 1rem;
}
.res-label { color: #C9D6F2; font-size: .9rem; font-weight: 500; }
.res-size  { color: #FFFFFF; font-size: 3.6rem; font-weight: 700; line-height: 1.05; letter-spacing: -0.02em; }
.res-conf  { color: #FFFFFF; font-size: 2.2rem; font-weight: 700; text-align: right; }
.res-right { text-align: right; }

/* Grafik batang */
.bar-row { display: flex; align-items: center; gap: .8rem; margin: .55rem 0; }
.bar-name { width: 2.6rem; color: var(--ink); font-weight: 600; font-size: .95rem; }
.bar-track { flex: 1; height: 10px; background: #E9EDF3; border-radius: 99px; overflow: hidden; }
.bar-fill { height: 100%; background: #9AA6B8; border-radius: 99px; }
.bar-row.on .bar-fill { background: var(--grad-flat); }
.bar-val { width: 3.6rem; text-align: right; color: var(--ink); font-size: .92rem; font-weight: 500; }
.bar-row.on .bar-name, .bar-row.on .bar-val { font-weight: 700; }
.badge {
    background: #0B0F1A; color: #FFFFFF; font-size: .72rem; font-weight: 600;
    padding: .15rem .55rem; border-radius: 99px;
}
.bar-row:not(.on) .badge { display: none; }

/* Tabel */
.tbl { width: 100%; border-collapse: separate; border-spacing: 0; font-size: .95rem;
       border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
.tbl th { background: #0B0F1A; color: #FFFFFF; text-align: left; padding: .7rem 1rem; font-weight: 600; }
.tbl td { background: #FFFFFF; color: var(--ink); padding: .7rem 1rem; border-top: 1px solid var(--line); }
.tbl tr.hl td { background: var(--grad-flat); color: #FFFFFF; font-weight: 700; border-top: 1px solid transparent; }

.note { color: var(--muted); font-size: .85rem; margin-top: .4rem; }

/* Expander */
[data-testid="stExpander"] { border: 1px solid var(--line) !important; border-radius: 14px !important; background: #FFFFFF !important; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color: var(--ink) !important; }
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
st.markdown(
    '<div class="hero"><div class="hero-title">Prediksi Ukuran Pakaian</div>'
    '<div class="hero-sub">Masukkan data tubuh Anda untuk mendapatkan rekomendasi ukuran '
    'berdasarkan standar EN 13402.</div></div>',
    unsafe_allow_html=True,
)

left, right = st.columns([5, 6], gap="large")

# ── Kolom kiri: input ────────────────────────────────────────────────────────
with left:
    with st.container(border=True):
        st.markdown('<div class="sec">Data tubuh</div>', unsafe_allow_html=True)

        gender = st.radio("Jenis kelamin", ["Pria", "Wanita"], horizontal=True)

        c1, c2 = st.columns(2)
        with c1:
            height = st.number_input("Tinggi badan (cm)", 100.0, 250.0, 170.0, 0.5)
        with c2:
            weight = st.number_input("Berat badan (kg)", 30.0, 200.0, 50.0, 0.5)

        bmi_now = weight / (height / 100) ** 2
        st.caption(f"BMI {bmi_now:.1f} — {bmi_category(bmi_now)}")

        st.markdown('<div class="sec" style="margin-top:.6rem">Lingkar tubuh</div>',
                    unsafe_allow_html=True)
        chest = st.number_input("Lingkar dada (cm)", 50.0, 200.0, 96.0, 0.5)
        st.caption("Ukur bagian terlebar dada, melingkar secara horizontal.")
        c3, c4 = st.columns(2)
        with c3:
            waist = st.number_input("Lingkar pinggang (cm)", 40.0, 200.0, 90.0, 0.5)
        with c4:
            hip = st.number_input("Lingkar pinggul (cm)", 50.0, 200.0, 100.0, 0.5)

        submitted = st.button("Prediksi ukuran", type="primary", use_container_width=True)

# ── Kolom kanan: hasil ───────────────────────────────────────────────────────
with right:
    if not submitted:
        st.markdown(
            '<div class="card"><div class="sec">Hasil prediksi</div>'
            '<div class="sec-sub">Isi data di sebelah kiri, lalu tekan tombol Prediksi ukuran.</div>'
            '<div class="note">Prediksi memakai data CAESAR dan standar EN 13402. '
            'Ukuran aktual tiap merek bisa berbeda, jadi gunakan hasilnya sebagai acuan.</div></div>',
            unsafe_allow_html=True,
        )
    else:
        gender_int = 0 if gender == "Pria" else 1
        size, conf, bmi_val, dist = do_predict(
            artefak, gender_int, height, weight, chest, waist, hip
        )
        k = artefak.get("k", "?")

        st.markdown(
            f'<div class="res"><div><div class="res-label">Ukuran yang disarankan</div>'
            f'<div class="res-size">{size}</div></div>'
            f'<div class="res-right"><div class="res-label">Tingkat keyakinan</div>'
            f'<div class="res-conf">{conf:.1f}%</div></div></div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="card"><div class="sec">Distribusi tetangga terdekat</div>'
            f'<div class="sec-sub">Proporsi ukuran dari {k} data terdekat</div>'
            f'{dist_bars(dist, size)}</div>',
            unsafe_allow_html=True,
        )

        col_header = "Lingkar dada" if gender == "Pria" else "Bust"
        rows = [
            [s, EN13402[gender][s], "Sesuai" if s == size else ""]
            for s in SIZE_ORDER
        ]
        hl = SIZE_ORDER.index(size) if size in SIZE_ORDER else None
        st.markdown(
            f'<div class="card"><div class="sec">Rentang EN 13402 untuk {gender}</div>'
            f'<div class="sec-sub">Baris berwarna biru adalah ukuran hasil prediksi.</div>'
            f'{html_table(["Ukuran", col_header, "Keterangan"], rows, hl)}'
            f'<div class="note">Prediksi berdasarkan data CAESAR (Amerika dan Eropa) serta '
            f'standar EN 13402. Ukuran aktual tiap merek pakaian bisa berbeda.</div></div>',
            unsafe_allow_html=True,
        )

# ── Info model ───────────────────────────────────────────────────────────────
st.write("")
with st.expander("Tentang model"):
    k = artefak.get("k", "?")
    wv = artefak.get("w_val", "?")
    acc = artefak.get("holdout_acc", 0) * 100
    f1 = artefak.get("holdout_f1", 0) * 100
    info = [
        ["Algoritma", "K-Nearest Neighbor (KNN)"],
        ["k (tetangga)", k],
        ["Bobot lingkar dada", f"×{wv}"],
        ["Jarak", "Euclidean"],
        ["Scaler", "MinMaxScaler"],
        ["Dataset", "CAESAR (4.435 sampel, 7 fitur)"],
        ["Akurasi hold-out", f"{acc:.2f}%"],
        ["F1-macro hold-out", f"{f1:.2f}%"],
    ]
    st.markdown(html_table(["Parameter", "Nilai"], info), unsafe_allow_html=True)

st.markdown(
    '<div class="note" style="margin-top:1.5rem;border-top:1px solid #E3E6EB;padding-top:1rem">'
    'Prediksi Ukuran Pakaian Menggunakan Algoritma K-Nearest Neighbor dengan Pembobotan '
    'Fitur Lingkar Dada pada Dataset CAESAR — Nafis Muyassar</div>',
    unsafe_allow_html=True,
)
