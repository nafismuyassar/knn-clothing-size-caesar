"""
app.py — Streamlit App: Prediksi Ukuran Pakaian (KNN + CAESAR)
Nafis Muyassar
"""
import pickle
import numpy as np
import streamlit as st

# ── Konstanta ────────────────────────────────────────────────────────────────
MODEL_PATH = "model/model.pkl"

EN13402_TABLE = {
    "Pria": {
        "XS": "78–86 cm", "S": "86–94 cm",  "M":  "94–102 cm",
        "L": "102–110 cm","XL":"110–118 cm", "XXL":"118–129 cm",
    },
    "Wanita": {
        "XS": "74–82 cm", "S": "82–90 cm",  "M":  "90–98 cm",
        "L":  "98–107 cm","XL":"107–119 cm", "XXL":"119–131 cm",
    },
}

SIZE_ORDER = ["XS", "S", "M", "L", "XL", "XXL"]

# ── Custom CSS ───────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
<style>
/* ── Font ── */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem !important; padding-bottom: 3rem !important; max-width: 680px !important; }

/* ── Background ── */
.stApp { background: #F8F8F7; }

/* ── Wordmark ── */
.wm { font-size: 22px; font-weight: 800; letter-spacing: -.5px; color: #111110; margin-bottom: 2px; }
.wm em { font-style: normal; color: #2563EB; }
.wm-sub { font-size: 13px; color: #A8A8A4; margin-bottom: 1.5rem; }

/* ── Section header ── */
h2, h3 { font-weight: 700 !important; letter-spacing: -.3px !important; color: #111110 !important; }

/* ── Input labels ── */
label { font-size: 11px !important; font-weight: 700 !important; letter-spacing: .05em !important;
        text-transform: uppercase !important; color: #6E6E6A !important; }

/* ── Number inputs ── */
.stNumberInput input {
    border: 1.5px solid #E2E2E0 !important;
    border-radius: 8px !important;
    background: #F3F3F2 !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    color: #111110 !important;
    padding: 8px 12px !important;
}
.stNumberInput input:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,.1) !important;
}

/* ── Radio ── */
.stRadio label { font-size: 14px !important; font-weight: 600 !important;
                 text-transform: none !important; letter-spacing: 0 !important; color: #111110 !important; }

/* ── Button ── */
.stButton > button {
    background: #2563EB !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    padding: 12px 20px !important;
    width: 100% !important;
    transition: background .15s !important;
}
.stButton > button:hover { background: #1D4ED8 !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    font-size: 12px !important; font-weight: 700 !important;
    color: #6E6E6A !important; letter-spacing: .04em !important; text-transform: uppercase !important;
    background: #F3F3F2 !important; border-radius: 8px !important; border: 1px solid #E2E2E0 !important;
}

/* ── Divider ── */
hr { border-color: #E2E2E0 !important; }

/* ── Metric (BMI) ── */
[data-testid="metric-container"] {
    background: #F3F3F2; border: 1px solid #E2E2E0;
    border-radius: 8px; padding: 10px 14px;
}
[data-testid="metric-container"] label {
    font-size: 10px !important; color: #A8A8A4 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 22px !important; font-weight: 800 !important; color: #111110 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ── Prediksi ─────────────────────────────────────────────────────────────────
def predict(artefak, gender_int, height_cm, weight_kg, chest_cm, waist_cm, hip_cm):
    bmi = weight_kg / (height_cm / 100) ** 2
    x   = np.array([[gender_int, height_cm, weight_kg, chest_cm,
                     waist_cm, hip_cm, bmi]], dtype=float)

    sc         = artefak["scaler"]
    knn        = artefak["knn"]
    feats      = artefak["feats"]
    w_feat     = artefak["w_feat"]
    w_val      = artefak["w_val"]
    id_to_size = artefak["id_to_size"]
    size_order = artefak["size_order"]

    x_s       = sc.transform(x).copy()
    x_s[:, feats.index(w_feat)] *= w_val

    pred_id   = knn.predict(x_s)[0]
    proba     = knn.predict_proba(x_s)[0]

    size_label = id_to_size[pred_id]
    confidence = float(proba[pred_id]) * 100
    dist       = {size_order[i]: float(p) * 100 for i, p in enumerate(proba)}
    return size_label, confidence, round(float(bmi), 1), dist


# ── Helpers ───────────────────────────────────────────────────────────────────
def bmi_label(bmi):
    if bmi < 18.5: return "Berat badan kurang"
    if bmi < 25:   return "Berat badan normal"
    if bmi < 30:   return "Kelebihan berat badan"
    return "Obesitas"

def render_result(size, conf, gender, dist, artefak):
    # ── Hero box ──
    st.markdown(f"""
<div style="background:#2563EB;border-radius:12px;padding:28px 24px 22px;
            display:flex;align-items:flex-end;justify-content:space-between;margin-bottom:12px;">
  <div>
    <div style="font-size:72px;font-weight:800;letter-spacing:-4px;line-height:1;color:#fff;">{size}</div>
    <div style="font-size:12px;color:rgba(255,255,255,.65);margin-top:4px;">
      Ukuran EN 13402 · {gender}
    </div>
  </div>
  <div style="text-align:right;">
    <div style="font-size:32px;font-weight:800;color:#fff;letter-spacing:-1px;line-height:1;">
      {conf:.0f}%
    </div>
    <div style="font-size:11px;color:rgba(255,255,255,.55);">kecocokan</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Distribusi ──
    st.markdown('<p style="font-size:10px;font-weight:700;letter-spacing:.06em;'
                'text-transform:uppercase;color:#A8A8A4;margin-bottom:8px;">Distribusi Tetangga</p>',
                unsafe_allow_html=True)

    size_order = artefak["size_order"]
    bars = ""
    for s in size_order:
        pct = dist.get(s, 0)
        is_hl = s == size
        clr   = "#2563EB" if is_hl else "#D1D1CE"
        lbl_c = "#2563EB" if is_hl else "#A8A8A4"
        fw    = "700" if is_hl else "500"
        bars += f"""
<div style="display:flex;align-items:center;gap:8px;margin-bottom:7px;">
  <span style="width:28px;font-size:11px;font-weight:{fw};color:{lbl_c};text-align:right;">{s}</span>
  <div style="flex:1;height:6px;background:#EBEBEA;border-radius:99px;overflow:hidden;">
    <div style="width:{pct:.1f}%;height:100%;background:{clr};border-radius:99px;"></div>
  </div>
  <span style="width:36px;font-size:11px;color:#A8A8A4;text-align:right;font-variant-numeric:tabular-nums;">{pct:.1f}%</span>
</div>"""
    st.markdown(f'<div style="margin-bottom:16px;">{bars}</div>', unsafe_allow_html=True)

    # ── Tabel EN 13402 ──
    st.markdown('<p style="font-size:10px;font-weight:700;letter-spacing:.06em;'
                'text-transform:uppercase;color:#A8A8A4;margin-bottom:8px;">'
                f'Rentang EN 13402 · {gender}</p>', unsafe_allow_html=True)

    tbl_data = EN13402_TABLE[gender]
    rows = ""
    for s in size_order:
        is_hl  = s == size
        bg     = "#EFF6FF" if is_hl else "transparent"
        s_clr  = "#2563EB" if is_hl else "#111110"
        v_clr  = "#111110" if is_hl else "#6E6E6A"
        fw     = "700" if is_hl else "400"
        badge  = (' <span style="font-size:9px;font-weight:700;background:#2563EB;color:#fff;'
                  'padding:1px 6px;border-radius:20px;">Kamu</span>') if is_hl else ""
        rows += f"""
<tr style="background:{bg};">
  <td style="padding:8px 10px;font-weight:700;color:{s_clr};border-bottom:1px solid #EBEBEA;">{s}{badge}</td>
  <td style="padding:8px 10px;font-weight:{fw};color:{v_clr};border-bottom:1px solid #EBEBEA;">{tbl_data.get(s,'-')}</td>
</tr>"""

    st.markdown(f"""
<div style="border:1px solid #E2E2E0;border-radius:10px;overflow:hidden;margin-bottom:12px;">
<table style="width:100%;border-collapse:collapse;font-size:13px;">
  <thead>
    <tr style="border-bottom:1px solid #E2E2E0;">
      <th style="padding:7px 10px;font-size:10px;font-weight:700;letter-spacing:.05em;
                 text-transform:uppercase;color:#A8A8A4;text-align:left;">Ukuran</th>
      <th style="padding:7px 10px;font-size:10px;font-weight:700;letter-spacing:.05em;
                 text-transform:uppercase;color:#A8A8A4;text-align:left;">
        {'Lingkar Dada' if gender == 'Pria' else 'Bust'}</th>
    </tr>
  </thead>
  <tbody>{rows}</tbody>
</table>
</div>
""", unsafe_allow_html=True)

    # ── Catatan ──
    st.markdown("""
<div style="background:#F3F3F2;border-radius:8px;padding:10px 13px;
            font-size:11px;color:#A8A8A4;line-height:1.6;">
  Hasil mengacu pada standar EN 13402 dan dataset CAESAR (Amerika/Eropa).
  Ukuran antar merek bisa berbeda — gunakan sebagai referensi awal.
</div>
""", unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="FitSize — Prediksi Ukuran Pakaian",
        page_icon="👕",
        layout="centered",
    )
    inject_css()

    # Wordmark
    st.markdown("""
<div class="wm">Fit<em>Size</em></div>
<div class="wm-sub">Prediksi ukuran pakaian berdasarkan ukuran tubuh &amp; standar EN 13402</div>
""", unsafe_allow_html=True)

    # Load model
    try:
        artefak = load_model()
    except FileNotFoundError:
        st.error("⚠️ File model tidak ditemukan (`model/model.pkl`). "
                 "Jalankan `python train_model.py` terlebih dahulu.")
        st.stop()

    # Info model (collapsed)
    with st.expander("INFO MODEL"):
        k   = artefak.get("k", "?")
        w   = artefak.get("w_val", "?")
        acc = artefak.get("holdout_acc", 0) * 100
        f1  = artefak.get("holdout_f1", 0) * 100
        st.markdown(f"""
| Parameter | Nilai |
|-----------|-------|
| Algoritma | K-Nearest Neighbor |
| k (tetangga) | {k} |
| Bobot lingkar dada | ×{w} |
| Jarak | Euclidean |
| Scaler | MinMaxScaler |
| Dataset | CAESAR (4.435 sampel) |
| Hold-out Accuracy | **{acc:.2f}%** |
| Hold-out F1-macro | **{f1:.2f}%** |
""")

    st.markdown("---")

    # ── Form ──
    col1, col2 = st.columns(2)
    with col1:
        gender = st.radio("Jenis Kelamin", ["Pria", "Wanita"], horizontal=True)
        height = st.number_input("Tinggi Badan (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.5)
        weight = st.number_input("Berat Badan (kg)",  min_value=30.0,  max_value=200.0, value=50.0,  step=0.5)
        chest  = st.number_input("Lingkar Dada / Bust (cm)", min_value=50.0, max_value=200.0, value=96.0, step=0.5)
    with col2:
        waist  = st.number_input("Lingkar Pinggang (cm)", min_value=40.0, max_value=200.0, value=90.0,  step=0.5)
        hip    = st.number_input("Lingkar Pinggul (cm)",  min_value=50.0, max_value=200.0, value=100.0, step=0.5)
        bmi_now = weight / (height / 100) ** 2
        st.metric("BMI (otomatis)", f"{bmi_now:.1f}")
        st.caption(bmi_label(bmi_now))

    st.markdown("---")

    if st.button("Lihat Ukuranku"):
        gender_int = 0 if gender == "Pria" else 1
        size, conf, bmi_val, dist = predict(
            artefak, gender_int, height, weight, chest, waist, hip)
        render_result(size, conf, gender, dist, artefak)

    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align:center;font-size:11px;color:#A8A8A4;line-height:1.8;">'
        'Prediksi Ukuran Pakaian Menggunakan Algoritma K-Nearest Neighbor '
        'dengan Pembobotan Fitur Lingkar Dada pada Dataset CAESAR</em><br>'
        '</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
