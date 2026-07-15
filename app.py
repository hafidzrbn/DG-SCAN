import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
from PIL import Image

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="DG-Scan: Deteksi Gizi & Rekomendasi Pintar",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tema Styling Kustom
st.markdown("""
    <style>
    .main {
        background-color: #0f1115;
        color: #e0e6ed;
    }
    .stAlert {
        border-radius: 8px;
    }
    .health-badge {
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        text-align: center;
    }
    .badge-healthy { background-color: #00c853; color: white; }
    .badge-medium { background-color: #ffc107; color: black; }
    .badge-risk { background-color: #f44336; color: white; }
    .badge-outlier { background-color: #2196f3; color: white; }

    /* Custom Premium Sidebar dark styling */
    [data-testid="stSidebar"] {
        background-color: #111315 !important;
        border-right: 1px solid #1a1d20 !important;
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: #e0e6ed;
    }
    
    /* Sidebar Section Title */
    .sidebar-section-title {
        color: #5d6778;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 25px;
        margin-bottom: 8px;
        padding-left: 5px;
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit radio button circle */
    div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child {
        display: none !important;
    }
    
    /* Style Streamlit radio labels to look like premium menu items */
    div[role="radiogroup"] label[data-baseweb="radio"] {
        background-color: transparent !important;
        border: none !important;
        padding: 12px 16px !important;
        border-radius: 12px !important;
        margin-bottom: 6px !important;
        color: #8a94a6 !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        cursor: pointer !important;
        transition: all 0.2s ease-in-out !important;
    }
    
    /* Radio item hover */
    div[role="radiogroup"] label[data-baseweb="radio"]:hover {
        background-color: #1c1e22 !important;
        color: #ffffff !important;
    }
    
    /* Radio item active/checked */
    div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) {
        background-color: #1e2125 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.15) !important;
        border-left: 4px solid #00c853 !important; /* Accent bar */
    }

    /* Style primary button to match reference (Active) */
    div.stButton > button[kind="primary"] {
        background-color: #2f6bf4 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
        box-shadow: 0px 4px 10px rgba(47, 107, 244, 0.2) !important;
    }
    div.stButton > button[kind="primary"]:hover, div.stButton > button[kind="primary"]:active {
        background-color: #1d4ed8 !important;
        color: #ffffff !important;
        box-shadow: 0px 4px 12px rgba(29, 78, 216, 0.3) !important;
    }

    /* Style secondary/default button to match reference (Active) */
    div.stButton > button {
        background-color: #e2e8f0 !important;
        color: #1e293b !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }
    div.stButton > button:hover, div.stButton > button:active {
        background-color: #cbd5e1 !important;
        color: #0f172a !important;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 1. CACHING LOAD MODEL & DATABASE
# ----------------------------------------------------
@st.cache_resource
def load_yolo_model():
    model_path = 'best.pt'
    if os.path.exists(model_path):
        return YOLO(model_path)
    return None

@st.cache_data
def load_recommendation_db():
    db_path = 'data/ggl_minuman_bukan_susu_rekomendasi.csv'
    if os.path.exists(db_path):
        return pd.read_csv(db_path)
    return None

# Load model dan dataset
model = load_yolo_model()
df_db = load_recommendation_db()

# Mapping Kelas YOLO
MAPPING_PRODUK = {
    0: "Golda Coffee Dolce Latte",
    1: "Nipis Madu Lime Soda",
    2: "Pocari Sweat",
    3: "Teh Pucuk Harum",
    4: "You C1000 Vitamin Orange"
}

# ----------------------------------------------------
# 2. LOGIKA SISTEM REKOMENDASI
# ----------------------------------------------------
def pindai_produk(nama_produk, df):
    produk = df[df['Nama Produk'].str.contains(nama_produk, case=False, na=False)]
    return produk.iloc[0] if not produk.empty else None

def rekomendasi_produk(produk_dipindai, df):
    kategori_dipindai = produk_dipindai['Kategori']
    klaster_dipindai = produk_dipindai['Klaster_DBSCAN']
    
    # Menentukan klaster target yang diizinkan (sama atau lebih sehat)
    if klaster_dipindai == 0:
        klaster_diizinkan = [0]
    elif klaster_dipindai == 1:
        klaster_diizinkan = [0, 1]
    elif klaster_dipindai == 2:
        klaster_diizinkan = [0, 1, 2]
    else:
        klaster_diizinkan = [0, 1, 2, -1]
        
    gula_dipindai = produk_dipindai['Gula (g) per 100 ml']
    lemak_dipindai = produk_dipindai['Lemak Total (g) per 100 ml']
    natrium_dipindai = produk_dipindai['Natrium (g) per 100 ml']
    
    # Filter kategori, klaster, dan bukan produk itu sendiri
    df_filter = df[
        (df['Kategori'] == kategori_dipindai) & 
        (df['Klaster_DBSCAN'].isin(klaster_diizinkan)) &
        (df['Nama Produk'] != produk_dipindai['Nama Produk'])
    ]

    # Filter kandungan gizi lebih rendah atau sama
    df_filter = df_filter[
        (df_filter['Gula (g) per 100 ml'] <= gula_dipindai) &
        (df_filter['Lemak Total (g) per 100 ml'] <= lemak_dipindai) &
        (df_filter['Natrium (g) per 100 ml'] <= natrium_dipindai)
    ]
    
    if df_filter.empty:
        return pd.DataFrame()

    features = ['Gula (g) per 100 ml_scaled', 'Lemak Total (g) per 100 ml_scaled', 'Natrium (g) per 100 ml_scaled']
    produk_dipindai_vector = produk_dipindai[features].values.reshape(1, -1)
    df_filter_vectors = df_filter[features].values
    
    # Hitung kemiripan kosinus
    cosine_sim = cosine_similarity(produk_dipindai_vector, df_filter_vectors).flatten()
    df_filter = df_filter.copy()
    df_filter['Kemiripan'] = cosine_sim
    
    # Urutkan berdasarkan gula terkecil dan kemiripan tertinggi
    df_filter = df_filter.sort_values(by=['Gula (g) per 100 ml', 'Kemiripan'], ascending=[True, False])
    return df_filter.head(5)

# ----------------------------------------------------
# 3. ANTARMUKA UTAMA (DASHBOARD)
# ----------------------------------------------------
st.title("🎯 DG-Scan: Deteksi Gizi & Rekomendasi Pintar")
st.markdown("Aplikasi pintar untuk memindai kemasan minuman, menganalisis profil risiko kesehatan, dan menyajikan opsi alternatif yang lebih sehat secara real-time.")

# Cek kesiapan model & database
if model is None or df_db is None:
    st.error("❌ ERROR: File `best.pt` atau database `data/ggl_minuman_bukan_susu_rekomendasi.csv` tidak ditemukan. Pastikan file berada di direktori proyek.")
    st.stop()

# Sidebar Pengaturan
# 2. Main Menu Section title and options
st.sidebar.markdown('<div class="sidebar-section-title">Menu Utama</div>', unsafe_allow_html=True)
mode_input = st.sidebar.radio(
    "Pilih Mode Deteksi", 
    ["📷 Kamera Live", "🔍 Pencarian Manual"],
    label_visibility="collapsed"
)

# 3. Static YOLO Threshold configuration
conf_threshold = 0.70

# State Manajemen untuk produk terpilih
if "selected_product" not in st.session_state:
    st.session_state.selected_product = None
if "temp_detected" not in st.session_state:
    st.session_state.temp_detected = None
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False

# ----------------------------------------------------
# 4. IMPLEMENTASI MODE DETEKSI
# ----------------------------------------------------
col_input, col_analysis = st.columns([1, 1])

with col_input:
    st.subheader("📸 Area Pemindaian")
    
    if mode_input == "📷 Kamera Live":
        # Toggle camera state using primary (blue) / secondary (grey) buttons matching design specs
        if not st.session_state.camera_active:
            if st.button("📷 Mulai Memindai", type="primary", key="start_cam"):
                st.session_state.camera_active = True
                # Clear previous temporary and selected products on new scan
                st.session_state.temp_detected = None
                st.rerun()
        else:
            if st.button("❌ Hentikan Kamera", type="secondary", key="stop_cam"):
                st.session_state.camera_active = False
                st.rerun()
                
        frame_placeholder = st.image([])
        
        # Tampilkan tombol Kunci Produk di bawah frame kamera jika kamera sedang aktif
        if st.session_state.camera_active:
            # Berikan sedikit jarak vertikal agar rapi
            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
            if st.button("🔒 Kunci Produk Terdeteksi", type="primary", key="lock_cam"):
                if st.session_state.get('temp_detected'):
                    st.session_state.selected_product = st.session_state.temp_detected
                    st.session_state.camera_active = False # Turn off camera automatically!
                    st.rerun()
                else:
                    st.warning("⚠️ Silakan tunggu sampai produk terdeteksi oleh kamera sebelum menekan Kunci.")
        
        if st.session_state.camera_active:
            cap = cv2.VideoCapture(0)
            
            while st.session_state.camera_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("Kamera gagal membaca gambar.")
                    break
                
                frame = cv2.flip(frame, 1)
                results = model(frame, verbose=False, conf=conf_threshold)
                
                annotated_frame = frame.copy()
                current_view = "Mencari..."
                status_color = (255, 255, 255) # Putih (Default)
                
                if len(results[0].boxes) > 0:
                    best_box = max(results[0].boxes, key=lambda x: x.conf[0])
                    cls_id = int(best_box.cls[0])
                    
                    if cls_id in MAPPING_PRODUK:
                        product_name = MAPPING_PRODUK[cls_id]
                        current_view = product_name
                        st.session_state.temp_detected = product_name
                        
                        # Cek klaster gizi di database untuk menentukan warna box
                        prod_row = df_db[df_db['Nama Produk'] == product_name]
                        if not prod_row.empty:
                            klaster = prod_row.iloc[0]['Klaster_DBSCAN']
                            if klaster == 0:
                                status_color = (0, 255, 0)      # Hijau
                            elif klaster == 1:
                                status_color = (0, 255, 255)    # Kuning
                            elif klaster == 2:
                                status_color = (0, 0, 255)      # Merah
                            else:
                                status_color = (255, 0, 0)      # Biru
                        
                        # Gambar bounding box dengan warna sesuai tingkat kesehatan
                        box = best_box.xyxy[0].cpu().numpy().astype(int)
                        cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), status_color, 3)
                        
                        # Tampilkan label produk
                        label_text = f"{product_name} ({best_box.conf[0]:.2f})"
                        cv2.rectangle(annotated_frame, (box[0], box[1] - 25), (box[0] + len(label_text)*10, box[1]), status_color, -1)
                        cv2.putText(annotated_frame, label_text, (box[0] + 5, box[1] - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
                
                # Tampilkan status target di pojok kiri atas frame
                cv2.putText(annotated_frame, f"Target: {current_view}", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2, cv2.LINE_AA)
                
                # Konversi frame BGR ke RGB untuk Streamlit
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb)
            
            cap.release()
            cv2.destroyAllWindows()
            
            # Auto-rerun to trigger layout change on successful manual lock
            if st.session_state.selected_product is not None and not st.session_state.camera_active:
                st.rerun()



    elif mode_input == "🔍 Pencarian Manual":
        nama_pilihan = st.selectbox("Pilih Produk Minuman:", MAPPING_PRODUK.values())
        if st.button("Pilih Produk"):
            st.session_state.selected_product = nama_pilihan
            st.rerun()

# ----------------------------------------------------
# 5. PEMBUATAN DETAIL ANALISIS & GRAFIK NUTRITIONAL
# ----------------------------------------------------
with col_analysis:
    st.subheader("📊 Hasil Analisis Kandungan Gizi")
    
    if st.session_state.selected_product:
        produk_dipindai = pindai_produk(st.session_state.selected_product, df_db)
        
        if produk_dipindai is not None:
            # Mengatur tampilan kartu gizi produk terpilih
            st.markdown(f"### **{produk_dipindai['Nama Produk']}**")
            st.text(f"Kategori: {produk_dipindai['Kategori']}")
            
            klaster = produk_dipindai['Klaster_DBSCAN']
            if klaster == 0:
                badge_html = '<span class="health-badge badge-healthy">🟢 KLASTER 0 (RISIKO RENDAH / SEHAT)</span>'
                rec_alert = st.success
            elif klaster == 1:
                badge_html = '<span class="health-badge badge-medium">🟡 KLASTER 1 (RISIKO SEDANG)</span>'
                rec_alert = st.warning
            elif klaster == 2:
                badge_html = '<span class="health-badge badge-risk">🔴 KLASTER 2 (RISIKO TINGGI)</span>'
                rec_alert = st.error
            else:
                badge_html = '<span class="health-badge badge-outlier">🔵 OUTLIER (KLASTER -1)</span>'
                rec_alert = st.info
                
            st.markdown(badge_html, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Metrik zat gizi utama (per 100 ml)
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Gula (per 100ml)", f"{produk_dipindai['Gula (g) per 100 ml']:.2f} g")
            m_col2.metric("Lemak Total (per 100ml)", f"{produk_dipindai['Lemak Total (g) per 100 ml']:.2f} g")
            m_col3.metric("Natrium (per 100ml)", f"{produk_dipindai['Natrium (g) per 100 ml']:.2f} g")
            
            # Hitung rekomendasi produk sehat
            rekomendasi = rekomendasi_produk(produk_dipindai, df_db)
            
            # Grafik Perbandingan Gizi menggunakan Bar Chart (Plotly)
            if not rekomendasi.empty:
                # Siapkan data untuk grafik perbandingan
                chart_data = pd.concat([pd.DataFrame([produk_dipindai]), rekomendasi], ignore_index=True)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name='Gula (g)',
                    x=chart_data['Nama Produk'],
                    y=chart_data['Gula (g) per 100 ml'],
                    marker_color='#ffc107'
                ))
                fig.add_trace(go.Bar(
                    name='Lemak (g)',
                    x=chart_data['Nama Produk'],
                    y=chart_data['Lemak Total (g) per 100 ml'],
                    marker_color='#00c853'
                ))
                fig.add_trace(go.Bar(
                    name='Natrium (g)',
                    x=chart_data['Nama Produk'],
                    y=chart_data['Natrium (g) per 100 ml'] * 10, # Skala natrium agar kelihatan di grafik
                    marker_color='#f44336',
                    hovertemplate='%{y:.2f} g (Skala x10)'
                ))
                
                fig.update_layout(
                    title="Perbandingan Kandungan Gizi per 100ml",
                    barmode='group',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#e0e6ed',
                    legend_title_text='Nutrisi',
                    xaxis_tickangle=-45,
                    height=350,
                    margin=dict(l=20, r=20, t=40, b=100)
                )
                st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.error("⚠️ Detail gizi produk tidak ditemukan di database.")
    else:
        st.info("💡 Silakan pindai kemasan produk atau cari produk secara manual untuk melihat analisis kandungan gizi.")

# ----------------------------------------------------
# 6. BAGIAN REKOMENDASI PRODUK
# ----------------------------------------------------
st.markdown("---")
st.subheader("💡 Rekomendasi Produk Alternatif Lebih Sehat")

if st.session_state.selected_product and 'rekomendasi' in locals() and not rekomendasi.empty:
    st.markdown("Berikut adalah opsi minuman sejenis yang memiliki nilai gizi (gula, garam/natrium, lemak) yang **lebih rendah atau setara**, diurutkan berdasarkan kesamaan kandungan dan tingkat kesehatan optimal:")
    
    # Tampilkan kartu visual untuk tiap produk rekomendasi
    rec_cols = st.columns(len(rekomendasi))
    for i, (_, row) in enumerate(rekomendasi.iterrows()):
        with rec_cols[i]:
            # Menentukan badge warna klaster rekomendasi
            rec_klaster = row['Klaster_DBSCAN']
            if rec_klaster == 0:
                box_border = "border-left: 5px solid #00c853;"
                badge_text = "🟢 Sehat"
            elif rec_klaster == 1:
                box_border = "border-left: 5px solid #ffc107;"
                badge_text = "🟡 Sedang"
            else:
                box_border = "border-left: 5px solid #f44336;"
                badge_text = "🔴 Risiko"
                
            st.markdown(f"""
                <div style="background-color: #1a1d24; padding: 15px; border-radius: 8px; {box_border} height: 180px;">
                    <h5 style="margin: 0; color: #fff; font-size: 14px;">{row['Nama Produk']}</h5>
                    <span style="font-size: 11px; color: #888;">{row['Kategori']}</span><br>
                    <span style="font-weight: bold; font-size: 12px; color: #fff;">{badge_text}</span>
                    <hr style="margin: 8px 0; border-color: #333;">
                    <span style="font-size: 11px; display: block;">Gula: {row['Gula (g) per 100 ml']:.2f} g</span>
                    <span style="font-size: 11px; display: block;">Lemak: {row['Lemak Total (g) per 100 ml']:.2f} g</span>
                    <span style="font-size: 11px; display: block;">Natrium: {row['Natrium (g) per 100 ml']:.2f} g</span>
                </div>
            """, unsafe_allow_html=True)
            
    # Tampilkan tabel data mentah pembanding
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Tabel Detail Perbandingan Alternatif Sehat:**")
    display_cols = ['Nama Produk', 'Kategori', 'Gula (g) per 100 ml', 'Lemak Total (g) per 100 ml', 'Natrium (g) per 100 ml', 'Klaster_DBSCAN', 'Kemiripan']
    st.dataframe(rekomendasi[display_cols].style.format({
        'Gula (g) per 100 ml': '{:.2f} g',
        'Lemak Total (g) per 100 ml': '{:.2f} g',
        'Natrium (g) per 100 ml': '{:.2f} g',
        'Kemiripan': '{:.4f}'
    }), use_container_width=True)

elif st.session_state.selected_product and 'rekomendasi' in locals() and rekomendasi.empty:
    st.success(f"✨ Hebat! **{produk_dipindai['Nama Produk']}** sudah merupakan pilihan minuman paling sehat dan rendah risiko di kategorinya.")
else:
    st.info("Pindai produk atau pilih secara manual untuk menampilkan alternatif rekomendasi gizi di sini.")
