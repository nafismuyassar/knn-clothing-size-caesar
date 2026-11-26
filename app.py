"""
app.py — Streamlit App: Prediksi Ukuran Pakaian (KNN + CAESAR)
Nafis Muyassar
"""
import pickle
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

MODEL_PATH = "model/model.pkl"
SIZE_ORDER = ["XS", "S", "M", "L", "XL", "XXL"]
EN13402 = {
    "Pria":   {"XS":"78–86","S":"86–94","M":"94–102","L":"102–110","XL":"110–118","XXL":"118–129"},
    "Wanita": {"XS":"74–82","S":"82–90","M":"90–98", "L":"98–107", "XL":"107–119","XXL":"119–131"},
}

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def do_predict(artefak, gender_int, height, weight, chest, waist, hip):
    bmi = weight / (height / 100) ** 2
    x   = np.array([[gender_int, height, weight, chest, waist, hip, bmi]], dtype=float)
    sc, knn = artefak["scaler"], artefak["knn"]
    feats, w_feat, w_val = artefak["feats"], artefak["w_feat"], artefak["w_val"]
    x_s = sc.transform(x).copy()
    x_s[:, feats.index(w_feat)] *= w_val
    pred_id = knn.predict(x_s)[0]
    proba   = knn.predict_proba(x_s)[0]
    size    = artefak["id_to_size"][pred_id]
    conf    = float(proba[pred_id]) * 100
    dist    = {artefak["size_order"][i]: float(p)*100 for i, p in enumerate(proba)}
    return size, conf, round(float(bmi), 1), dist

def bmi_info(b):
    if b < 18.5: return "Kurang",  "#1D4ED8", "#DBEAFE"
    if b < 25:   return "Normal",  "#065F46", "#D1FAE5"
    if b < 30:   return "Lebih",   "#92400E", "#FEF3C7"
    return               "Obesitas","#991B1B", "#FEE2E2"

def main():
    st.set_page_config(page_title="FitSize", page_icon="👕", layout="centered")

    # ── inject full-page custom HTML via components ──
    # We use st.markdown for CSS reset + render everything via HTML components

    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Plus Jakarta Sans',sans-serif!important;}
#MainMenu,footer,header{visibility:hidden!important;}
.block-container{padding:2rem 1rem 4rem!important;max-width:500px!important;}
.stApp{background:#F8F8F7!important;}
/* hide streamlit number input arrows styling, keep clean */
button[kind="primary"]{display:none}
div[data-testid="stVerticalBlock"]{gap:0!important;}
</style>
""", unsafe_allow_html=True)

    # load model
    try:
        artefak = load_model()
    except FileNotFoundError:
        st.error("Model tidak ditemukan. Jalankan `python train_model.py` dulu.")
        st.stop()

    # ── STATE ──
    if "result" not in st.session_state:
        st.session_state.result = None

    # ── RENDER FORM via HTML + collect via st widgets hidden ──
    # Wordmark
    st.markdown("""
<div style="margin-bottom:20px;">
  <div style="font-size:22px;font-weight:800;letter-spacing:-.5px;color:#111110;">
    Fit<span style="color:#2563EB;">Size</span>
  </div>
  <div style="font-size:12px;color:#A8A8A4;margin-top:3px;">
    Prediksi ukuran pakaian berdasarkan ukuran tubuh &amp; standar EN&nbsp;13402
  </div>
</div>
<div style="background:#fff;border:1px solid #E2E2E0;border-radius:12px;padding:20px;margin-bottom:10px;">
  <div style="font-size:10px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;
              color:#A8A8A4;margin-bottom:14px;">Ukuran Tubuh</div>
""", unsafe_allow_html=True)

    # Gender
    gender = st.radio("Jenis Kelamin", ["Pria", "Wanita"], horizontal=True,
                      label_visibility="collapsed")
    st.markdown(f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-bottom:14px;">
  <div onclick="this.dataset.v='Pria'" style="padding:10px;border:1.5px solid {'#2563EB' if gender=='Pria' else '#E2E2E0'};
       border-radius:8px;background:{'#EFF6FF' if gender=='Pria' else '#F3F3F2'};
       color:{'#2563EB' if gender=='Pria' else '#6E6E6A'};font-size:13px;font-weight:600;
       display:flex;align-items:center;justify-content:center;gap:6px;">
    ♂ Pria
  </div>
  <div style="padding:10px;border:1.5px solid {'#2563EB' if gender=='Wanita' else '#E2E2E0'};
       border-radius:8px;background:{'#EFF6FF' if gender=='Wanita' else '#F3F3F2'};
       color:{'#2563EB' if gender=='Wanita' else '#6E6E6A'};font-size:13px;font-weight:600;
       display:flex;align-items:center;justify-content:center;gap:6px;">
    ♀ Wanita
  </div>
</div>
""", unsafe_allow_html=True)

    # inputs — 2 col
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<p style="font-size:11px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;color:#6E6E6A;margin-bottom:4px;">Tinggi</p>', unsafe_allow_html=True)
        height = st.number_input("Tinggi", 100.0, 250.0, 170.0, 0.5, label_visibility="collapsed", key="h")
        st.markdown('<p style="font-size:11px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;color:#6E6E6A;margin:10px 0 4px;">Berat</p>', unsafe_allow_html=True)
        weight = st.number_input("Berat", 30.0, 200.0, 50.0, 0.5, label_visibility="collapsed", key="w")
        st.markdown('<p style="font-size:11px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;color:#6E6E6A;margin:10px 0 4px;">Lingkar Dada</p>', unsafe_allow_html=True)
        chest  = st.number_input("Dada", 50.0, 200.0, 96.0, 0.5, label_visibility="collapsed", key="c")
    with c2:
        st.markdown('<p style="font-size:11px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;color:#6E6E6A;margin-bottom:4px;">Lingkar Pinggang</p>', unsafe_allow_html=True)
        waist  = st.number_input("Pinggang", 40.0, 200.0, 90.0, 0.5, label_visibility="collapsed", key="ws")
        st.markdown('<p style="font-size:11px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;color:#6E6E6A;margin:10px 0 4px;">Lingkar Pinggul</p>', unsafe_allow_html=True)
        hip    = st.number_input("Pinggul", 50.0, 200.0, 100.0, 0.5, label_visibility="collapsed", key="hp")
        # BMI display
        bmi_now = weight / (height/100)**2
        bmi_lbl, bmi_tc, bmi_bg = bmi_info(bmi_now)
        st.markdown(f"""
<div style="margin-top:10px;background:#F3F3F2;border:1px solid #E2E2E0;border-radius:8px;padding:10px 12px;">
  <div style="font-size:10px;font-weight:700;letter-spacing:.05em;text-transform:uppercase;color:#A8A8A4;">BMI</div>
  <div style="font-size:22px;font-weight:800;color:#111110;font-variant-numeric:tabular-nums;">{bmi_now:.1f}</div>
  <div style="display:inline-block;font-size:10px;font-weight:700;padding:2px 8px;border-radius:20px;
              background:{bmi_bg};color:{bmi_tc};margin-top:2px;">{bmi_lbl}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # close panel

    # ── CTA button ──
    st.markdown("""
<style>
div[data-testid="stButton"] > button {
  background:#2563EB!important;color:#fff!important;
  border:none!important;border-radius:10px!important;
  font-size:14px!important;font-weight:700!important;
  padding:13px 20px!important;width:100%!important;
  font-family:'Plus Jakarta Sans',sans-serif!important;
  margin-top:4px!important;
}
div[data-testid="stButton"] > button:hover{background:#1D4ED8!important;}
div[data-testid="stButton"] > button:focus{box-shadow:none!important;outline:none!important;}
</style>
""", unsafe_allow_html=True)

    if st.button("Lihat Ukuranku", use_container_width=True):
        gender_int = 0 if gender == "Pria" else 1
        size, conf, bmi_val, dist = do_predict(artefak, gender_int, height, weight, chest, waist, hip)
        st.session_state.result = (size, conf, gender, dist)

    # ── RESULT ──
    if st.session_state.result:
        size, conf, gdr, dist = st.session_state.result

        # hero
        st.markdown(f"""
<div style="background:#2563EB;border-radius:12px;padding:28px 24px 22px;
            display:flex;align-items:flex-end;justify-content:space-between;margin-top:12px;">
  <div>
    <div style="font-size:72px;font-weight:800;letter-spacing:-4px;line-height:1;color:#fff;">{size}</div>
    <div style="font-size:12px;color:rgba(255,255,255,.65);margin-top:6px;">Ukuran EN 13402 · {gdr}</div>
  </div>
  <div style="text-align:right;">
    <div style="font-size:36px;font-weight:800;color:#fff;letter-spacing:-1px;line-height:1;">{conf:.0f}%</div>
    <div style="font-size:11px;color:rgba(255,255,255,.55);">kecocokan</div>
  </div>
</div>
""", unsafe_allow_html=True)

        # panel bawah
        st.markdown('<div style="background:#fff;border:1px solid #E2E2E0;border-top:none;border-radius:0 0 12px 12px;padding:18px;">', unsafe_allow_html=True)

        # distribusi
        st.markdown('<div style="font-size:10px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:#A8A8A4;margin-bottom:10px;">Distribusi</div>', unsafe_allow_html=True)
        bars = ""
        for s in SIZE_ORDER:
            pct  = dist.get(s, 0)
            hl   = s == size
            clr  = "#2563EB" if hl else "#D1D1CE"
            lc   = "#2563EB" if hl else "#A8A8A4"
            fw   = "700"     if hl else "500"
            bars += f"""<div style="display:flex;align-items:center;gap:8px;margin-bottom:7px;">
  <span style="width:28px;font-size:11px;font-weight:{fw};color:{lc};text-align:right;">{s}</span>
  <div style="flex:1;height:6px;background:#EBEBEA;border-radius:99px;overflow:hidden;">
    <div style="width:{pct:.1f}%;height:100%;background:{clr};border-radius:99px;"></div>
  </div>
  <span style="width:36px;font-size:11px;color:#A8A8A4;text-align:right;font-variant-numeric:tabular-nums;">{pct:.1f}%</span>
</div>"""
        st.markdown(f'<div style="margin-bottom:16px;">{bars}</div>', unsafe_allow_html=True)

        # tabel EN 13402
        col_lbl = "Lingkar Dada (cm)" if gdr == "Pria" else "Bust (cm)"
        st.markdown(f'<div style="font-size:10px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:#A8A8A4;margin-bottom:8px;">Rentang EN 13402 · {gdr}</div>', unsafe_allow_html=True)
        tbl_data = EN13402[gdr]
        rows = ""
        for s in SIZE_ORDER:
            hl    = s == size
            bg    = "#EFF6FF" if hl else "transparent"
            sc2   = "#2563EB" if hl else "#111110"
            vc    = "#111110" if hl else "#6E6E6A"
            fw    = "700"     if hl else "400"
            badge = ('<span style="font-size:9px;font-weight:700;background:#2563EB;color:#fff;'
                     'padding:1px 6px;border-radius:20px;margin-left:5px;">Kamu</span>') if hl else ""
            rows += f"""<tr style="background:{bg};">
  <td style="padding:8px 10px;font-weight:700;color:{sc2};border-bottom:1px solid #EBEBEA;">{s}{badge}</td>
  <td style="padding:8px 10px;font-weight:{fw};color:{vc};border-bottom:1px solid #EBEBEA;">{tbl_data.get(s,'-')} cm</td>
</tr>"""
        st.markdown(f"""
<div style="border:1px solid #E2E2E0;border-radius:10px;overflow:hidden;margin-bottom:14px;">
<table style="width:100%;border-collapse:collapse;font-size:13px;">
<thead><tr style="border-bottom:1px solid #E2E2E0;">
  <th style="padding:7px 10px;font-size:10px;font-weight:700;letter-spacing:.05em;text-transform:uppercase;color:#A8A8A4;text-align:left;">Ukuran</th>
  <th style="padding:7px 10px;font-size:10px;font-weight:700;letter-spacing:.05em;text-transform:uppercase;color:#A8A8A4;text-align:left;">{col_lbl}</th>
</tr></thead>
<tbody>{rows}</tbody>
</table></div>
""", unsafe_allow_html=True)

        # catatan
        st.markdown("""
<div style="background:#F3F3F2;border-radius:8px;padding:10px 13px;font-size:11px;color:#A8A8A4;line-height:1.6;">
  Hasil mengacu pada standar EN 13402 dan dataset CAESAR (Amerika/Eropa).
  Ukuran antar merek bisa berbeda — gunakan sebagai referensi awal.
</div>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)  # close res-body

    # footer
    st.markdown("""
<div style="text-align:center;font-size:11px;color:#A8A8A4;line-height:1.8;margin-top:24px;">
  <em>Prediksi Ukuran Pakaian Menggunakan Algoritma K-Nearest Neighbor
  dengan Pembobotan Fitur Lingkar Dada pada Dataset CAESAR</em>
</div>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
