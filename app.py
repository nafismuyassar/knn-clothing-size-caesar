"""
app.py — Streamlit App: Prediksi Ukuran Pakaian (KNN + CAESAR)
Nafis Muyassar
"""
import pickle, numpy as np, streamlit as st

MODEL_PATH = "model/model.pkl"
SIZE_ORDER = ["XS","S","M","L","XL","XXL"]
EN13402 = {
    0: {"XS":"78–86","S":"86–94","M":"94–102","L":"102–110","XL":"110–118","XXL":"118–129"},
    1: {"XS":"74–82","S":"82–90","M":"90–98", "L":"98–107", "XL":"107–119","XXL":"119–131"},
}

@st.cache_resource
def load_model():
    with open(MODEL_PATH,"rb") as f: return pickle.load(f)

def do_predict(artefak, gender_int, height, weight, chest, waist, hip):
    bmi = weight/(height/100)**2
    x   = np.array([[gender_int,height,weight,chest,waist,hip,bmi]],dtype=float)
    sc,knn = artefak["scaler"],artefak["knn"]
    x_s = sc.transform(x).copy()
    x_s[:,artefak["feats"].index(artefak["w_feat"])] *= artefak["w_val"]
    pid   = knn.predict(x_s)[0]
    proba = knn.predict_proba(x_s)[0]
    size  = artefak["id_to_size"][pid]
    conf  = float(proba[pid])*100
    dist  = {artefak["size_order"][i]:float(p)*100 for i,p in enumerate(proba)}
    return size, conf, round(float(bmi),1), dist

# ─────────────────────────────────────────────
st.set_page_config(page_title="FitSize",page_icon="👕",layout="centered")

try:
    artefak = load_model()
except FileNotFoundError:
    st.error("Model tidak ditemukan. Jalankan `python train_model.py` dulu.")
    st.stop()

# ── Sembunyikan SEMUA chrome Streamlit, paksa background putih bersih ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
*{font-family:'Plus Jakarta Sans',sans-serif!important;box-sizing:border-box;}
#MainMenu,footer,header,[data-testid="stToolbar"]{display:none!important;}
.stApp,.stApp>div,.block-container{background:#F8F8F7!important;}
.block-container{max-width:480px!important;padding:36px 16px 60px!important;}
/* sembunyikan semua widget bawaan — kita render sendiri via HTML */
[data-testid="stForm"],[data-testid="stVerticalBlock"]>div:has(>[data-testid="stRadio"]),
[data-testid="stRadio"]{display:none!important;}
/* khusus number input & button: sembunyikan label bawaan */
[data-testid="stNumberInput"] label,
[data-testid="stButton"] {display:none!important;}
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE defaults ──
def ss(k,v):
    if k not in st.session_state: st.session_state[k]=v
ss("gender",0); ss("height",170.0); ss("weight",50.0)
ss("chest",96.0); ss("waist",90.0); ss("hip",100.0)
ss("result",None)

# ── Hidden real Streamlit widgets (untuk nilai form) ──
# Kita butuh ini supaya st.button bisa trigger rerun
_g  = st.radio("g",["Pria","Wanita"],index=st.session_state.gender,label_visibility="collapsed")
_h  = st.number_input("h", value=st.session_state.height, label_visibility="collapsed", key="_h")
_w  = st.number_input("w", value=st.session_state.weight, label_visibility="collapsed", key="_w")
_c  = st.number_input("c", value=st.session_state.chest,  label_visibility="collapsed", key="_c")
_ws = st.number_input("ws",value=st.session_state.waist,  label_visibility="collapsed", key="_ws")
_hp = st.number_input("hp",value=st.session_state.hip,    label_visibility="collapsed", key="_hp")

# Simpan ke session state
st.session_state.gender = 0 if _g=="Pria" else 1
st.session_state.height = _h; st.session_state.weight = _w
st.session_state.chest  = _c; st.session_state.waist  = _ws; st.session_state.hip = _hp

g   = st.session_state.gender
h,w,c,ws,hp = _h,_w,_c,_ws,_hp
bmi = w/(h/100)**2

def bmi_chip(b):
    if b<18.5: return "Kurang","#1E40AF","#DBEAFE"
    if b<25:   return "Normal","#065F46","#D1FAE5"
    if b<30:   return "Lebih", "#92400E","#FEF3C7"
    return             "Obesitas","#991B1B","#FEE2E2"
blbl,btc,bbg = bmi_chip(bmi)

# ── Render UI HTML penuh ──
gp_on  = "#2563EB" if g==0 else "#E2E2E0"
gp_bg  = "#EFF6FF" if g==0 else "#F3F3F2"
gp_tc  = "#2563EB" if g==0 else "#6E6E6A"
gw_on  = "#2563EB" if g==1 else "#E2E2E0"
gw_bg  = "#EFF6FF" if g==1 else "#F3F3F2"
gw_tc  = "#2563EB" if g==1 else "#6E6E6A"

st.markdown(f"""
<div style="font-size:22px;font-weight:800;letter-spacing:-.5px;color:#111110;margin-bottom:3px;">
  Fit<span style="color:#2563EB;">Size</span>
</div>
<div style="font-size:12px;color:#A8A8A4;margin-bottom:20px;">
  Prediksi ukuran pakaian berdasarkan ukuran tubuh &amp; standar EN&nbsp;13402
</div>

<div style="background:#fff;border:1px solid #E2E2E0;border-radius:12px;padding:20px 20px 16px;margin-bottom:10px;">

  <div style="font-size:10px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;
              color:#A8A8A4;margin-bottom:14px;">Jenis Kelamin</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:18px;">
    <div style="padding:11px 8px;border:1.5px solid {gp_on};border-radius:8px;background:{gp_bg};
                color:{gp_tc};font-size:13px;font-weight:700;text-align:center;">♂&nbsp; Pria</div>
    <div style="padding:11px 8px;border:1.5px solid {gw_on};border-radius:8px;background:{gw_bg};
                color:{gw_tc};font-size:13px;font-weight:700;text-align:center;">♀&nbsp; Wanita</div>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">

    <div>
      <div style="font-size:10px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;
                  color:#A8A8A4;margin-bottom:6px;">Tinggi</div>
      <div style="background:#F3F3F2;border:1.5px solid #E2E2E0;border-radius:8px;
                  padding:10px 12px;font-size:16px;font-weight:600;color:#111110;
                  display:flex;justify-content:space-between;align-items:center;">
        <span>{h:.1f}</span><span style="font-size:11px;color:#A8A8A4;font-weight:500;">cm</span>
      </div>
    </div>

    <div>
      <div style="font-size:10px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;
                  color:#A8A8A4;margin-bottom:6px;">Lingkar Pinggang</div>
      <div style="background:#F3F3F2;border:1.5px solid #E2E2E0;border-radius:8px;
                  padding:10px 12px;font-size:16px;font-weight:600;color:#111110;
                  display:flex;justify-content:space-between;align-items:center;">
        <span>{ws:.1f}</span><span style="font-size:11px;color:#A8A8A4;font-weight:500;">cm</span>
      </div>
    </div>

    <div>
      <div style="font-size:10px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;
                  color:#A8A8A4;margin-bottom:6px;">Berat</div>
      <div style="background:#F3F3F2;border:1.5px solid #E2E2E0;border-radius:8px;
                  padding:10px 12px;font-size:16px;font-weight:600;color:#111110;
                  display:flex;justify-content:space-between;align-items:center;">
        <span>{w:.1f}</span><span style="font-size:11px;color:#A8A8A4;font-weight:500;">kg</span>
      </div>
    </div>

    <div>
      <div style="font-size:10px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;
                  color:#A8A8A4;margin-bottom:6px;">Lingkar Pinggul</div>
      <div style="background:#F3F3F2;border:1.5px solid #E2E2E0;border-radius:8px;
                  padding:10px 12px;font-size:16px;font-weight:600;color:#111110;
                  display:flex;justify-content:space-between;align-items:center;">
        <span>{hp:.1f}</span><span style="font-size:11px;color:#A8A8A4;font-weight:500;">cm</span>
      </div>
    </div>

    <div>
      <div style="font-size:10px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;
                  color:#A8A8A4;margin-bottom:6px;">Lingkar Dada</div>
      <div style="background:#F3F3F2;border:1.5px solid #E2E2E0;border-radius:8px;
                  padding:10px 12px;font-size:16px;font-weight:600;color:#111110;
                  display:flex;justify-content:space-between;align-items:center;">
        <span>{c:.1f}</span><span style="font-size:11px;color:#A8A8A4;font-weight:500;">cm</span>
      </div>
    </div>

    <div>
      <div style="font-size:10px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;
                  color:#A8A8A4;margin-bottom:6px;">BMI</div>
      <div style="background:#F3F3F2;border:1.5px solid #E2E2E0;border-radius:8px;
                  padding:10px 12px;display:flex;align-items:center;justify-content:space-between;">
        <span style="font-size:18px;font-weight:800;color:#111110;font-variant-numeric:tabular-nums;">{bmi:.1f}</span>
        <span style="font-size:10px;font-weight:700;padding:3px 8px;border-radius:20px;
                     background:{bbg};color:{btc};">{blbl}</span>
      </div>
    </div>

  </div>
</div>

<p style="font-size:11px;color:#A8A8A4;margin:0 0 8px;">
  Ubah nilai di input di bawah lalu klik <strong>Lihat Ukuranku</strong>
</p>
""", unsafe_allow_html=True)

# ── Widget number input yang TERLIHAT (tidak disembunyikan) ──
st.markdown("""
<style>
/* Override Streamlit number input jadi clean */
[data-testid="stNumberInput"]{margin-bottom:0!important;}
[data-testid="stNumberInput"] label{
  font-size:10px!important;font-weight:700!important;
  letter-spacing:.07em!important;text-transform:uppercase!important;
  color:#A8A8A4!important;margin-bottom:4px!important;
  font-family:'Plus Jakarta Sans',sans-serif!important;
}
[data-testid="stNumberInput"] input{
  background:#F3F3F2!important;
  border:1.5px solid #E2E2E0!important;
  border-radius:8px!important;
  color:#111110!important;
  font-size:15px!important;font-weight:600!important;
  font-family:'Plus Jakarta Sans',sans-serif!important;
  padding:10px 12px!important;
}
[data-testid="stNumberInput"] input:focus{
  border-color:#2563EB!important;
  box-shadow:0 0 0 3px rgba(37,99,235,.12)!important;
  outline:none!important;
}
[data-testid="stNumberInput"] [data-testid="stNumberInputStepDown"],
[data-testid="stNumberInput"] [data-testid="stNumberInputStepUp"]{
  background:#E2E2E0!important;border:none!important;
  color:#111110!important;border-radius:6px!important;
  font-weight:700!important;
}
/* Radio sembunyikan karena sudah ada HTML di atas */
[data-testid="stRadio"]{display:none!important;}
/* Button */
[data-testid="stButton"]>button{
  display:flex!important;
  background:#2563EB!important;color:#fff!important;
  border:none!important;border-radius:10px!important;
  font-size:15px!important;font-weight:700!important;
  padding:13px!important;width:100%!important;
  font-family:'Plus Jakarta Sans',sans-serif!important;
  margin-top:8px!important;justify-content:center!important;
  cursor:pointer!important;
}
[data-testid="stButton"]>button:hover{background:#1D4ED8!important;}
[data-testid="stButton"]>button:focus{box-shadow:none!important;outline:none!important;}
/* sembunyikan label bawaan number input di sini juga */
</style>
""", unsafe_allow_html=True)

col1,col2 = st.columns(2)
with col1:
    st.number_input("Tinggi (cm)",    100.0,250.0,h,  0.5,key="hi")
    st.number_input("Berat (kg)",     30.0, 200.0,w,  0.5,key="wi")
    st.number_input("Lingkar Dada (cm)",50.0,200.0,c, 0.5,key="ci")
with col2:
    st.number_input("Lingkar Pinggang (cm)",40.0,200.0,ws,0.5,key="wsi")
    st.number_input("Lingkar Pinggul (cm)", 50.0,200.0,hp,0.5,key="hpi")

# Ambil nilai terkini
fh  = st.session_state.get("hi", h)
fw  = st.session_state.get("wi", w)
fc  = st.session_state.get("ci", c)
fws = st.session_state.get("wsi",ws)
fhp = st.session_state.get("hpi",hp)

if st.button("Lihat Ukuranku", use_container_width=True):
    size,conf,bmi_v,dist = do_predict(artefak, g, fh, fw, fc, fws, fhp)
    st.session_state.result = (size,conf,g,dist)

# ── HASIL ──
if st.session_state.result:
    size,conf,gdr,dist = st.session_state.result
    gdr_lbl = "Pria" if gdr==0 else "Wanita"
    col_lbl = "Lingkar Dada (cm)" if gdr==0 else "Bust (cm)"

    st.markdown(f"""
<div style="background:#2563EB;border-radius:12px;padding:28px 24px 22px;
            display:flex;align-items:flex-end;justify-content:space-between;margin-top:12px;">
  <div>
    <div style="font-size:80px;font-weight:800;letter-spacing:-4px;line-height:1;color:#fff;">{size}</div>
    <div style="font-size:12px;color:rgba(255,255,255,.65);margin-top:6px;">
      Ukuran EN&nbsp;13402 · {gdr_lbl}
    </div>
  </div>
  <div style="text-align:right;">
    <div style="font-size:36px;font-weight:800;color:#fff;letter-spacing:-1px;line-height:1;">{conf:.0f}%</div>
    <div style="font-size:11px;color:rgba(255,255,255,.55);">kecocokan</div>
  </div>
</div>
<div style="background:#fff;border:1px solid #E2E2E0;border-top:none;
            border-radius:0 0 12px 12px;padding:18px;margin-bottom:10px;">
""", unsafe_allow_html=True)

    # distribusi
    st.markdown('<div style="font-size:10px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:#A8A8A4;margin-bottom:10px;">Distribusi</div>', unsafe_allow_html=True)
    bars=""
    for s in SIZE_ORDER:
        pct=dist.get(s,0); hl=s==size
        clr="#2563EB" if hl else "#D1D1CE"
        lc="#2563EB"  if hl else "#A8A8A4"
        fw2="700"     if hl else "500"
        bars+=f"""<div style="display:flex;align-items:center;gap:8px;margin-bottom:7px;">
<span style="width:28px;font-size:11px;font-weight:{fw2};color:{lc};text-align:right;">{s}</span>
<div style="flex:1;height:6px;background:#EBEBEA;border-radius:99px;overflow:hidden;">
  <div style="width:{pct:.1f}%;height:100%;background:{clr};border-radius:99px;"></div>
</div>
<span style="width:36px;font-size:11px;color:#A8A8A4;text-align:right;font-variant-numeric:tabular-nums;">{pct:.1f}%</span>
</div>"""
    st.markdown(f'<div style="margin-bottom:16px;">{bars}</div>', unsafe_allow_html=True)

    # tabel
    tbl=EN13402[gdr]
    st.markdown(f'<div style="font-size:10px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:#A8A8A4;margin-bottom:8px;">Rentang EN&nbsp;13402 · {gdr_lbl}</div>', unsafe_allow_html=True)
    rows=""
    for s in SIZE_ORDER:
        hl=s==size
        bg="#EFF6FF" if hl else "transparent"
        sc2="#2563EB" if hl else "#111110"
        vc="#111110"  if hl else "#6E6E6A"
        fw3="700"     if hl else "400"
        badge=('<span style="font-size:9px;font-weight:700;background:#2563EB;color:#fff;'
               'padding:1px 6px;border-radius:20px;margin-left:5px;">Kamu</span>') if hl else ""
        rows+=f"""<tr style="background:{bg};">
<td style="padding:8px 10px;font-weight:700;color:{sc2};border-bottom:1px solid #EBEBEA;">{s}{badge}</td>
<td style="padding:8px 10px;font-weight:{fw3};color:{vc};border-bottom:1px solid #EBEBEA;">{tbl.get(s,'-')} cm</td>
</tr>"""
    st.markdown(f"""
<div style="border:1px solid #E2E2E0;border-radius:10px;overflow:hidden;margin-bottom:14px;">
<table style="width:100%;border-collapse:collapse;font-size:13px;">
<thead><tr><th style="padding:7px 10px;font-size:10px;font-weight:700;letter-spacing:.05em;
text-transform:uppercase;color:#A8A8A4;text-align:left;border-bottom:1px solid #E2E2E0;">Ukuran</th>
<th style="padding:7px 10px;font-size:10px;font-weight:700;letter-spacing:.05em;
text-transform:uppercase;color:#A8A8A4;text-align:left;border-bottom:1px solid #E2E2E0;">{col_lbl}</th>
</tr></thead><tbody>{rows}</tbody></table></div>
<div style="background:#F3F3F2;border-radius:8px;padding:10px 13px;font-size:11px;color:#A8A8A4;line-height:1.6;">
  Hasil mengacu pada standar EN&nbsp;13402 dan dataset CAESAR (Amerika/Eropa).
  Ukuran antar merek bisa berbeda — gunakan sebagai referensi awal.
</div>
</div>
""", unsafe_allow_html=True)

# footer
st.markdown("""
<div style="text-align:center;font-size:11px;color:#A8A8A4;line-height:1.8;margin-top:20px;">
  <em>Prediksi Ukuran Pakaian Menggunakan Algoritma K-Nearest Neighbor
  dengan Pembobotan Fitur Lingkar Dada pada Dataset CAESAR</em><br>
</div>
""", unsafe_allow_html=True)
