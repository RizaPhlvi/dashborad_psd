import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Dashboard Komoditas Perkebunan Indonesia",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# PROFESSIONAL CSS - TABLEAU/POWER BI STYLE
# =========================================================
st.markdown("""
<style>
    /* =========================================================
       GLOBAL - PROFESSIONAL DARK THEME
    ========================================================= */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
        color: #e8eaed;
    }

    .block-container {
        max-width: 1600px;
        padding: 1rem 2rem 3rem 2rem;
    }

    /* Hide default Streamlit menu */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* =========================================================
       SIDEBAR - PROFESSIONAL FILTER PANEL
    ========================================================= */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1429 0%, #1a1f3a 100%);
        border-right: 2px solid #2d3561;
        padding: 1rem 0.5rem;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 0.5rem;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    section[data-testid="stSidebar"] .stMarkdown {
        color: #cbd5e1 !important;
    }

    /* Filter section styling */
    .filter-section {
        background: rgba(45, 53, 97, 0.3);
        border: 1px solid rgba(100, 116, 139, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .filter-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }

    /* =========================================================
       HERO SECTION - PROFESSIONAL HEADER
    ========================================================= */
    .hero-container {
        background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 50%, #0f766e 100%);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 20px 60px rgba(30, 58, 138, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }

    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        border-radius: 50%;
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.5rem;
        letter-spacing: -0.03em;
        position: relative;
        z-index: 1;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: rgba(255, 255, 255, 0.9);
        line-height: 1.6;
        margin-bottom: 1rem;
        position: relative;
        z-index: 1;
    }

    .hero-badges {
        display: flex;
        gap: 0.75rem;
        flex-wrap: wrap;
        position: relative;
        z-index: 1;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.5rem 1rem;
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 100px;
        font-size: 0.875rem;
        font-weight: 600;
        color: #ffffff;
    }

    /* =========================================================
       SECTION HEADERS
    ========================================================= */
    .page-title {
        font-size: 1.875rem;
        font-weight: 800;
        color: #ffffff;
        margin-top: 0.5rem;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }

    .page-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }

    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #ffffff;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3b82f6;
        display: inline-block;
    }

    /* =========================================================
       KPI CARDS - PROFESSIONAL METRICS
    ========================================================= */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 32px rgba(59, 130, 246, 0.4);
        border-color: rgba(59, 130, 246, 0.6);
    }

    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.875rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
    }

    div[data-testid="stMetricDelta"] {
        color: #10b981 !important;
        font-weight: 600 !important;
    }

    /* =========================================================
       INFO/SUCCESS/WARNING BOXES
    ========================================================= */
    .insight-card {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(99, 102, 241, 0.1) 100%);
        border-left: 4px solid #3b82f6;
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.2);
        color: #dbeafe;
        line-height: 1.7;
    }

    .success-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(52, 211, 153, 0.1) 100%);
        border-left: 4px solid #10b981;
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(16, 185, 129, 0.2);
        color: #d1fae5;
        line-height: 1.7;
    }

    .warning-card {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(251, 191, 36, 0.1) 100%);
        border-left: 4px solid #f59e0b;
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(245, 158, 11, 0.2);
        color: #fef3c7;
        line-height: 1.7;
    }

    /* =========================================================
       TABS - PROFESSIONAL NAVIGATION
    ========================================================= */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(30, 41, 59, 0.5);
        padding: 0.5rem;
        border-radius: 12px;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        color: #cbd5e1;
        border: 1px solid transparent;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(59, 130, 246, 0.1);
        border-color: rgba(59, 130, 246, 0.3);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }

    /* =========================================================
       DATAFRAMES - CLEAN TABLES
    ========================================================= */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(59, 130, 246, 0.2);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }

    /* =========================================================
       BUTTONS
    ========================================================= */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.625rem 1.5rem;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.5);
    }

    /* =========================================================
       INPUTS
    ========================================================= */
    .stSelectbox label,
    .stSlider label,
    .stRadio label,
    .stMultiSelect label,
    .stNumberInput label {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* =========================================================
       EXPANDERS
    ========================================================= */
    div[data-testid="stExpander"] {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 12px;
        padding: 0.5rem;
    }

    /* =========================================================
       FOOTER
    ========================================================= */
    .professional-footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px;
        border: 1px solid rgba(59, 130, 246, 0.2);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }

    .professional-footer h3 {
        color: #ffffff;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .professional-footer p {
        color: #94a3b8;
        font-size: 0.9rem;
        margin: 0.25rem 0;
    }

    /* =========================================================
       CHART CONTAINERS
    ========================================================= */
    .chart-container {
        background: rgba(30, 41, 59, 0.3);
        border: 1px solid rgba(59, 130, 246, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }

    .chart-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 1rem;
        letter-spacing: -0.01em;
    }

    /* =========================================================
       RESPONSIVE ADJUSTMENTS
    ========================================================= */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 1.875rem;
        }
        
        .page-title {
            font-size: 1.5rem;
        }
        
        div[data-testid="stMetricValue"] {
            font-size: 1.5rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# DATASET EMBEDDED
# =========================================================
CSV_DATA = """Provinsi,Kelapa Sawit,Kelapa,Karet,Kopi,Kakao,Teh,Tebu
ACEH,1092.71,64.1,51.17,74.13,34.17,0,0
SUMATERA UTARA,5120.02,103.64,251.52,91.69,38.48,10.14,14.52
SUMATERA BARAT,1410.64,79.06,99.67,15.32,34.58,5.6,0
RIAU,9136.1,399.95,180.74,1.83,0.84,0,0
JAMBI,2113.97,115.35,222.56,23.23,0.69,4.24,0
SUMATERA SELATAN,3967.39,61.65,619.55,219.59,4.4,2.62,124.37
BENGKULU,1298.17,7.25,64.6,55.63,2.36,1.88,0
LAMPUNG,375.24,83.76,84.83,120.38,48.11,0,644.48
KEP. BANGKA BELITUNG,860.24,4.55,26.04,0.11,0.29,0,0
KEP. RIAU,24.19,12.16,8.41,0,0.01,0,0
DKI JAKARTA,0,0,0,0,0,0,0
JAWA BARAT,46.75,88.14,21.17,26.48,0.71,80.24,55.17
JAWA TENGAH,0,139.01,20.73,26.79,1.31,11.81,250.07
DI YOGYAKARTA,0,50.08,0.01,1.88,2.03,0.23,3.22
JAWA TIMUR,0,194.14,16.27,53.25,10.56,2.14,1252.84
BANTEN,32.84,45.83,5.77,2.02,2.12,0.01,0
BALI,0,68.39,0,14.71,4.87,0,0
NUSA TENGGARA BARAT,0,49.81,0,6.42,2.59,0,18.22
NUSA TENGGARA TIMUR,0,62.15,0,24.38,21.08,0,10.77
KALIMANTAN BARAT,4958.54,76.8,158.41,2.98,0.6,0,0
KALIMANTAN TENGAH,7458.14,16.6,108.29,0.24,1.61,0,0
KALIMANTAN SELATAN,1255.08,24.01,127.96,0.89,0.05,0,0
KALIMANTAN TIMUR,3905.19,9.82,53.49,0.12,1.03,0,0
KALIMANTAN UTARA,611.14,0.64,0.16,0.11,0.79,0,0
SULAWESI UTARA,0,270.3,0,3.72,5.62,0,0
SULAWESI TENGAH,384.2,199.64,1.37,3.13,125.2,0,0
SULAWESI SELATAN,133.5,67.57,3.77,31.79,80.52,0,22.56
SULAWESI TENGGARA,75.47,42.87,0.22,2.61,98.02,0,15.41
GORONTALO,22.83,66.7,0,0.13,1.54,0,53.89
SULAWESI BARAT,366.68,36.72,0,4.75,67.14,0,0
MALUKU,22.33,107.73,0.6,0.49,8.59,0,0
MALUKU UTARA,20.14,204.17,0,0.02,7.38,0,0
PAPUA BARAT,40.38,2.02,0,0.01,0.24,0,0
PAPUA BARAT DAYA,43.44,14.35,0,0,0.76,0,0
PAPUA,130.55,9.77,0,0.09,8,0,0
PAPUA SELATAN,482.37,4.35,4.82,0,0.01,0,0
PAPUA TENGAH,47.98,1.2,0,1.18,0.82,0,0
PAPUA PEGUNUNGAN,0,0.02,0,3.27,0,0,0
"""

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():
    return pd.read_csv(io.StringIO(CSV_DATA))

df = load_data()
numeric_cols = [c for c in df.columns if c != "Provinsi"]

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def format_num(x):
    try:
        return f"{float(x):,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
    except:
        return str(x)

def get_filtered_df(base_df, province_filter, show_zero_rows):
    data = base_df.copy()
    if province_filter != "Semua Provinsi":
        data = data[data["Provinsi"] == province_filter].copy()
    if not show_zero_rows:
        data = data[(data[numeric_cols].sum(axis=1) > 0)].copy()
    return data

def add_total_production(data):
    temp = data.copy()
    temp["Total Produksi"] = temp[numeric_cols].sum(axis=1)
    return temp

def top_province_for_commodity(data, commodity):
    if data.empty or commodity not in data.columns:
        return "-", 0
    idx = data[commodity].idxmax()
    return data.loc[idx, "Provinsi"], data.loc[idx, commodity]

def province_with_highest_total(data):
    temp = add_total_production(data)
    if temp.empty:
        return "-", 0
    idx = temp["Total Produksi"].idxmax()
    return temp.loc[idx, "Provinsi"], temp.loc[idx, "Total Produksi"]

def generate_dynamic_insight(data, commodity):
    if data.empty:
        return "Tidak ada data yang tersedia untuk filter saat ini."
    total = data[commodity].sum()
    if total <= 0:
        return f"Komoditas <b>{commodity}</b> tidak memiliki produksi pada filter aktif."
    top_df = data.nlargest(3, commodity)[["Provinsi", commodity]].copy()
    top_df["Share"] = (top_df[commodity] / total) * 100
    top1 = top_df.iloc[0]
    msg = (
        f"Produksi <b>{commodity}</b> pada filter aktif mencapai <b>{format_num(total)} ribu ton</b>. "
        f"Kontributor terbesar adalah <b>{top1['Provinsi']}</b> dengan produksi <b>{format_num(top1[commodity])} ribu ton</b> "
        f"atau sekitar <b>{top1['Share']:.1f}%</b> dari total produksi komoditas ini."
    )
    if len(top_df) >= 3:
        share3 = top_df["Share"].sum()
        msg += f" Tiga provinsi teratas menyumbang sekitar <b>{share3:.1f}%</b> dari total produksi."
    return msg

def generate_recommendations(data, commodity):
    if data.empty:
        return ["Tidak ada data yang tersedia untuk menghasilkan rekomendasi."]
    total_by_commodity = data[numeric_cols].sum().sort_values(ascending=False)
    dominant = total_by_commodity.index[0]
    dominant_val = total_by_commodity.iloc[0]
    top_prov, top_val = top_province_for_commodity(data, commodity)
    total_selected = data[commodity].sum()
    share = (top_val / total_selected * 100) if total_selected > 0 else 0
    return [
        f"Fokuskan strategi hilirisasi pada <b>{dominant}</b> karena memiliki total produksi tertinggi sebesar <b>{format_num(dominant_val)} ribu ton</b>.",
        f"Untuk komoditas <b>{commodity}</b>, perlu mitigasi risiko konsentrasi wilayah karena <b>{top_prov}</b> menyumbang sekitar <b>{share:.1f}%</b> dari total produksi.",
        "Pengembangan dashboard berikutnya sebaiknya menambahkan data historis multi-tahun agar forecasting menjadi lebih kuat secara metodologis.",
        "Integrasi peta spasial provinsi berbasis GeoJSON akan memperkuat kualitas storytelling geografis.",
        "Model machine learning di dashboard ini lebih tepat dibaca sebagai eksplorasi pola awal, bukan prediksi kebijakan final tanpa validasi tambahan."
    ]

def export_csv(dataframe):
    return dataframe.to_csv(index=False).encode("utf-8")

# =========================================================
# PLOTLY PROFESSIONAL TEMPLATE
# =========================================================
def apply_professional_layout(fig, height=500):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(30, 41, 59, 0.3)",
        plot_bgcolor="rgba(30, 41, 59, 0.3)",
        font=dict(color="#e8eaed", size=12),
        height=height,
        margin=dict(l=50, r=40, t=70, b=50),
        xaxis=dict(
            gridcolor="rgba(148, 163, 184, 0.1)",
            zerolinecolor="rgba(148, 163, 184, 0.2)",
            title_font=dict(size=13, weight=600)
        ),
        yaxis=dict(
            gridcolor="rgba(148, 163, 184, 0.1)",
            zerolinecolor="rgba(148, 163, 184, 0.2)",
            title_font=dict(size=13, weight=600)
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(148, 163, 184, 0.2)",
            font=dict(size=11)
        ),
        title=dict(
            font=dict(size=16, weight=700, color="#ffffff"),
            x=0.5,
            xanchor="center"
        )
    )
    return fig

# =========================================================
# HERO HEADER
# =========================================================
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🌾 Dashboard Komoditas Perkebunan Indonesia</div>
    <div class="hero-subtitle">
        Platform analitik interaktif untuk eksplorasi, visualisasi, dan pemodelan data produksi komoditas perkebunan Indonesia per provinsi. 
        Dirancang dengan pendekatan executive dashboard modern untuk kebutuhan akademik dan presentasi profesional.
    </div>
    <div class="hero-badges">
        <span class="hero-badge">📊 Executive Dashboard</span>
        <span class="hero-badge">🔍 Interactive EDA</span>
        <span class="hero-badge">🤖 ML Playground</span>
        <span class="hero-badge">📥 Export Ready</span>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR - PROFESSIONAL FILTER PANEL
# =========================================================
with st.sidebar:
    st.markdown("## 🧭 Navigasi")
    
    menu = st.radio(
        "Pilih Halaman",
        [
            "🏠 Executive Dashboard",
            "📊 Data Explorer",
            "🔍 EDA Explorer",
            "🧭 Profil Provinsi & Komoditas",
            "🗺️ Geo Insight",
            "📈 Predictive Analytics",
            "🧠 Insight & Rekomendasi",
            "📥 Export Center"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("## 🎛️ Filter Global")
    
    with st.container():
        st.markdown('<div class="filter-section">', unsafe_allow_html=True)
        
        st.markdown('<div class="filter-title">Komoditas Utama</div>', unsafe_allow_html=True)
        selected_commodity = st.selectbox(
            "Komoditas",
            numeric_cols,
            index=0,
            label_visibility="collapsed"
        )
        
        st.markdown('<div class="filter-title">Provinsi</div>', unsafe_allow_html=True)
        province_options = ["Semua Provinsi"] + df["Provinsi"].tolist()
        selected_province = st.selectbox(
            "Provinsi",
            province_options,
            index=0,
            label_visibility="collapsed"
        )
        
        st.markdown('<div class="filter-title">Top N Provinsi</div>', unsafe_allow_html=True)
        top_n = st.slider(
            "Jumlah",
            min_value=5,
            max_value=20,
            value=10,
            label_visibility="collapsed"
        )
        
        show_zero_rows = st.checkbox(
            "Tampilkan provinsi dengan total produksi 0",
            value=True
        )
        
        view_mode = st.radio(
            "Mode Tampilan",
            ["Nilai Absolut", "Persentase (%)"],
            index=0
        )
        
        st.markdown('</div>', unsafe_allow_html=True)

filtered_df = get_filtered_df(df, selected_province, show_zero_rows)

# =========================================================
# EXECUTIVE DASHBOARD
# =========================================================
if menu == "🏠 Executive Dashboard":
    st.markdown('<div class="page-title">Executive Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Ringkasan eksekutif kondisi perkebunan nasional berdasarkan filter aktif</div>', unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("Tidak ada data yang cocok dengan filter aktif.")
    else:
        # KPI Cards Row
        total_production = filtered_df[numeric_cols].sum().sum()
        total_province = filtered_df.shape[0]
        commodity_totals = filtered_df[numeric_cols].sum().sort_values(ascending=False)
        top_commodity = commodity_totals.index[0]
        top_commodity_val = commodity_totals.iloc[0]
        best_prov, best_total = province_with_highest_total(filtered_df)

        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Jumlah Provinsi", total_province, "Provinsi aktif")
        with col2:
            st.metric("Total Produksi", format_num(total_production), "Ribu ton")
        with col3:
            st.metric("Komoditas Dominan", top_commodity, format_num(top_commodity_val))
        with col4:
            st.metric("Provinsi Tertinggi", best_prov[:15], format_num(best_total))

        st.markdown("---")

        # Main Charts Row
        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown('<div class="section-header">Top Provinsi berdasarkan Komoditas Terpilih</div>', unsafe_allow_html=True)
            top_df = filtered_df[["Provinsi", selected_commodity]].sort_values(
                selected_commodity, ascending=False
            ).head(min(top_n, len(filtered_df))).copy()

            if view_mode == "Persentase (%)":
                total_val = top_df[selected_commodity].sum()
                top_df["Display"] = (top_df[selected_commodity] / total_val * 100) if total_val > 0 else 0
                x_col = "Display"
                x_title = "Persentase (%)"
            else:
                x_col = selected_commodity
                x_title = "Produksi (Ribu Ton)"

            fig = px.bar(
                top_df.sort_values(x_col, ascending=True),
                x=x_col,
                y="Provinsi",
                orientation="h",
                text=x_col,
                title=f"Top {min(top_n, len(top_df))} Provinsi - {selected_commodity}"
            )
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_layout(xaxis_title=x_title, yaxis_title="")
            fig = apply_professional_layout(fig, 540)
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.markdown('<div class="section-header">Komposisi Produksi Antar Komoditas</div>', unsafe_allow_html=True)
            comp_df = filtered_df[numeric_cols].sum().reset_index()
            comp_df.columns = ["Komoditas", "Produksi"]
            if view_mode == "Persentase (%)":
                total_comp = comp_df["Produksi"].sum()
                comp_df["Produksi"] = (comp_df["Produksi"] / total_comp * 100) if total_comp > 0 else 0

            fig2 = px.pie(
                comp_df,
                values="Produksi",
                names="Komoditas",
                hole=0.5,
                title="Distribusi Produksi per Komoditas"
            )
            fig2.update_traces(textinfo='percent+label')
            fig2 = apply_professional_layout(fig2, 540)
            st.plotly_chart(fig2, use_container_width=True)

        # Quick Insight
        st.markdown('<div class="section-header">Quick Insight</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="insight-card">{generate_dynamic_insight(filtered_df, selected_commodity)}</div>',
            unsafe_allow_html=True
        )

        # Summary Table
        st.markdown('<div class="section-header">Ringkasan Produksi per Provinsi</div>', unsafe_allow_html=True)
        summary_df = add_total_production(filtered_df)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

# =========================================================
# DATA EXPLORER
# =========================================================
elif menu == "📊 Data Explorer":
    st.markdown('<div class="page-title">Data Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Eksplorasi struktur, kualitas, dan statistik deskriptif dataset</div>', unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("Tidak ada data untuk filter aktif.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Jumlah Observasi", filtered_df.shape[0])
        with col2:
            st.metric("Jumlah Variabel", filtered_df.shape[1])
        with col3:
            st.metric("Missing Value", int(filtered_df.isnull().sum().sum()))

        st.markdown("---")

        tab1, tab2, tab3 = st.tabs(["📋 Dataset", "📈 Statistik Deskriptif", "🧹 Kualitas Data"])

        with tab1:
            st.markdown('<div class="section-header">Data Lengkap (Filtered)</div>', unsafe_allow_html=True)
            st.dataframe(filtered_df, use_container_width=True, hide_index=True)

        with tab2:
            st.markdown('<div class="section-header">Statistik Deskriptif</div>', unsafe_allow_html=True)
            desc = filtered_df[numeric_cols].describe().T
            desc["range"] = desc["max"] - desc["min"]
            st.dataframe(desc, use_container_width=True)

        with tab3:
            st.markdown('<div class="section-header">Laporan Kualitas Data</div>', unsafe_allow_html=True)
            duplicate_count = filtered_df.duplicated().sum()
            zero_rows = (filtered_df[numeric_cols].sum(axis=1) == 0).sum()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Missing Values", int(filtered_df.isnull().sum().sum()))
            with col2:
                st.metric("Baris Duplikat", int(duplicate_count))
            with col3:
                st.metric("Provinsi Produksi 0", int(zero_rows))

            st.markdown(
                '<div class="warning-card">Outlier tidak otomatis dihapus karena bisa merepresentasikan sentra produksi riil, bukan kesalahan data.</div>',
                unsafe_allow_html=True
            )

# =========================================================
# EDA EXPLORER
# =========================================================
elif menu == "🔍 EDA Explorer":
    st.markdown('<div class="page-title">EDA Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Eksplorasi visual interaktif dengan berbagai jenis grafik</div>', unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("Tidak ada data untuk filter aktif.")
    else:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📌 Overview",
            "📊 Distribution",
            "🔗 Relationship",
            "🔥 Correlation",
            "📍 Province Deep Dive"
        ])

        # TAB 1 - OVERVIEW
        with tab1:
            col1, col2 = st.columns([3, 2])

            with col1:
                st.markdown('<div class="section-header">Top Provinsi - Komoditas Terpilih</div>', unsafe_allow_html=True)
                top_df = filtered_df[["Provinsi", selected_commodity]].sort_values(
                    selected_commodity, ascending=False
                ).head(min(top_n, len(filtered_df))).copy()

                fig = px.bar(
                    top_df.sort_values(selected_commodity, ascending=True),
                    x=selected_commodity,
                    y="Provinsi",
                    orientation="h",
                    text=selected_commodity,
                    title=f"Top {min(top_n, len(top_df))} Provinsi - {selected_commodity}"
                )
                fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                fig = apply_professional_layout(fig, 500)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown('<div class="section-header">Top Provinsi - Total Produksi</div>', unsafe_allow_html=True)
                temp = add_total_production(filtered_df)
                top_total = temp.sort_values("Total Produksi", ascending=False).head(min(top_n, len(temp)))

                fig2 = px.bar(
                    top_total,
                    x="Provinsi",
                    y="Total Produksi",
                    text="Total Produksi",
                    title="Top Provinsi berdasarkan Total Produksi"
                )
                fig2.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                fig2.update_xaxes(tickangle=45)
                fig2 = apply_professional_layout(fig2, 500)
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown(
                f'<div class="insight-card">{generate_dynamic_insight(filtered_df, selected_commodity)}</div>',
                unsafe_allow_html=True
            )

        # TAB 2 - DISTRIBUTION
        with tab2:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="section-header">Histogram Distribusi</div>', unsafe_allow_html=True)
                fig = px.histogram(
                    filtered_df,
                    x=selected_commodity,
                    nbins=15,
                    title=f"Distribusi {selected_commodity}"
                )
                fig = apply_professional_layout(fig, 480)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown('<div class="section-header">Boxplot Perbandingan</div>', unsafe_allow_html=True)
                melted = filtered_df.melt(
                    id_vars="Provinsi",
                    value_vars=numeric_cols,
                    var_name="Komoditas",
                    value_name="Produksi"
                )

                fig2 = px.box(
                    melted,
                    x="Komoditas",
                    y="Produksi",
                    color="Komoditas",
                    title="Perbandingan Persebaran Antar Komoditas"
                )
                fig2.update_xaxes(tickangle=35)
                fig2.update_layout(showlegend=False)
                fig2 = apply_professional_layout(fig2, 480)
                st.plotly_chart(fig2, use_container_width=True)

        # TAB 3 - RELATIONSHIP
        with tab3:
            st.markdown('<div class="section-header">Analisis Hubungan Dua Variabel</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                x_var = st.selectbox("Variabel X", numeric_cols, index=0, key="eda_x")
            with col2:
                y_default = 1 if len(numeric_cols) > 1 else 0
                y_var = st.selectbox("Variabel Y", numeric_cols, index=y_default, key="eda_y")

            if x_var == y_var:
                st.warning("Pilih dua variabel yang berbeda untuk analisis hubungan.")
            else:
                fig = px.scatter(
                    filtered_df,
                    x=x_var,
                    y=y_var,
                    hover_name="Provinsi",
                    title=f"Hubungan {x_var} vs {y_var}"
                )
                fig = apply_professional_layout(fig, 550)
                st.plotly_chart(fig, use_container_width=True)

                corr_val = filtered_df[[x_var, y_var]].corr().iloc[0, 1]
                if pd.notna(corr_val):
                    st.markdown(
                        f'<div class="insight-card">Nilai korelasi Pearson antara <b>{x_var}</b> dan <b>{y_var}</b> adalah <b>{corr_val:.3f}</b>.</div>',
                        unsafe_allow_html=True
                    )

        # TAB 4 - CORRELATION
        with tab4:
            st.markdown('<div class="section-header">Heatmap Korelasi Antar Komoditas</div>', unsafe_allow_html=True)
            
            corr = filtered_df[numeric_cols].corr()

            fig = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                zmin=-1,
                zmax=1,
                color_continuous_scale=[
                    [0.0, "#dc2626"],
                    [0.5, "#1e293b"],
                    [1.0, "#059669"]
                ],
                title="Matriks Korelasi"
            )

            fig.update_traces(
                textfont=dict(color="white", size=12),
                xgap=3,
                ygap=3
            )

            fig.update_xaxes(side="bottom", tickangle=30)
            fig.update_yaxes(autorange="reversed")

            fig = apply_professional_layout(fig, 600)
            st.plotly_chart(fig, use_container_width=True)

        # TAB 5 - PROVINCE DEEP DIVE
        with tab5:
            st.markdown('<div class="section-header">Analisis Mendalam per Provinsi</div>', unsafe_allow_html=True)
            
            province_pick = st.selectbox(
                "Pilih provinsi",
                filtered_df["Provinsi"].tolist(),
                key="deep_prov"
            )
            p_df = filtered_df[filtered_df["Provinsi"] == province_pick]

            if not p_df.empty:
                row = p_df.iloc[0]
                profile = pd.DataFrame({
                    "Komoditas": numeric_cols,
                    "Produksi": [row[c] for c in numeric_cols]
                }).sort_values("Produksi", ascending=False)

                col1, col2 = st.columns([3, 2])

                with col1:
                    fig = px.bar(
                        profile,
                        x="Komoditas",
                        y="Produksi",
                        text="Produksi",
                        title=f"Profil Produksi {province_pick}"
                    )
                    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                    fig = apply_professional_layout(fig, 500)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    total_p = profile["Produksi"].sum()
                    dom_comm = profile.iloc[0]["Komoditas"]
                    dom_val = profile.iloc[0]["Produksi"]

                    st.metric("Total Produksi Provinsi", format_num(total_p))
                    st.metric("Komoditas Dominan", dom_comm, format_num(dom_val))

                    fig2 = px.pie(
                        profile,
                        names="Komoditas",
                        values="Produksi",
                        hole=0.45,
                        title="Komposisi Komoditas"
                    )
                    fig2.update_traces(textinfo='percent+label')
                    fig2 = apply_professional_layout(fig2, 420)
                    st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# PROFIL PROVINSI & KOMODITAS
# =========================================================
elif menu == "🧭 Profil Provinsi & Komoditas":
    st.markdown('<div class="page-title">Profil Provinsi & Komoditas</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Analisis mendalam untuk provinsi dan komoditas tertentu</div>', unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("Tidak ada data untuk filter aktif.")
    else:
        tab_p, tab_k = st.tabs(["📍 Profil Provinsi", "🌾 Profil Komoditas"])

        with tab_p:
            st.markdown('<div class="section-header">Pilih Provinsi</div>', unsafe_allow_html=True)
            province_pick = st.selectbox("Provinsi", filtered_df["Provinsi"].tolist(), key="profile_province", label_visibility="collapsed")
            p_df = filtered_df[filtered_df["Provinsi"] == province_pick]

            if not p_df.empty:
                row = p_df.iloc[0]
                prov_profile = pd.DataFrame({
                    "Komoditas": numeric_cols,
                    "Produksi": [row[c] for c in numeric_cols]
                }).sort_values("Produksi", ascending=False)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Produksi", format_num(prov_profile["Produksi"].sum()))
                with col2:
                    st.metric("Komoditas Dominan", prov_profile.iloc[0]["Komoditas"])
                with col3:
                    st.metric("Nilai Tertinggi", format_num(prov_profile.iloc[0]["Produksi"]))

                fig = px.bar(
                    prov_profile,
                    x="Komoditas",
                    y="Produksi",
                    text="Produksi",
                    title=f"Struktur Komoditas - {province_pick}"
                )
                fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                fig = apply_professional_layout(fig, 520)
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(prov_profile, use_container_width=True, hide_index=True)

        with tab_k:
            st.markdown('<div class="section-header">Pilih Komoditas</div>', unsafe_allow_html=True)
            commodity_pick = st.selectbox("Komoditas", numeric_cols, key="profile_commodity", label_visibility="collapsed")
            cdf = filtered_df[["Provinsi", commodity_pick]].sort_values(commodity_pick, ascending=False).copy()
            total = cdf[commodity_pick].sum()

            top_prov, top_val = top_province_for_commodity(filtered_df, commodity_pick)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Nasional", format_num(total))
            with col2:
                st.metric("Provinsi Tertinggi", top_prov)
            with col3:
                st.metric("Produksi Tertinggi", format_num(top_val))

            fig = px.bar(
                cdf.head(min(top_n, len(cdf))),
                x="Provinsi",
                y=commodity_pick,
                text=commodity_pick,
                title=f"Top {min(top_n, len(cdf))} Provinsi - {commodity_pick}"
            )
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_xaxes(tickangle=45)
            fig = apply_professional_layout(fig, 520)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                f'<div class="insight-card">{generate_dynamic_insight(filtered_df, commodity_pick)}</div>',
                unsafe_allow_html=True
            )

# =========================================================
# GEO INSIGHT
# =========================================================
elif menu == "🗺️ Geo Insight":
    st.markdown('<div class="page-title">Geo Insight</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Visualisasi persebaran spasial produksi komoditas</div>', unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("Tidak ada data untuk filter aktif.")
    else:
        geo_df = filtered_df[["Provinsi", selected_commodity]].copy().sort_values(selected_commodity, ascending=False)

        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown('<div class="section-header">Ranking Spasial Produksi</div>', unsafe_allow_html=True)
            fig = px.bar(
                geo_df.head(min(top_n, len(geo_df))),
                x="Provinsi",
                y=selected_commodity,
                color=selected_commodity,
                text=selected_commodity,
                title=f"Ranking Spasial Produksi {selected_commodity}"
            )
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_xaxes(tickangle=45)
            fig.update_layout(showlegend=False)
            fig = apply_professional_layout(fig, 520)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="section-header">Bubble Geo-Insight</div>', unsafe_allow_html=True)
            pseudo_geo = geo_df.head(min(15, len(geo_df))).copy()
            pseudo_geo["Rank"] = range(1, len(pseudo_geo) + 1)

            fig2 = px.scatter(
                pseudo_geo,
                x="Rank",
                y=selected_commodity,
                size=selected_commodity,
                hover_name="Provinsi",
                text="Provinsi",
                title="Bubble Chart Prioritas Wilayah"
            )
            fig2.update_traces(textposition="top center")
            fig2 = apply_professional_layout(fig2, 520)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown(
            f'<div class="insight-card">Komoditas <b>{selected_commodity}</b> memperlihatkan konsentrasi produksi pada sejumlah provinsi utama. Versi premium berikutnya bisa ditingkatkan menjadi <b>choropleth map Indonesia</b> berbasis GeoJSON.</div>',
            unsafe_allow_html=True
        )

# =========================================================
# PREDICTIVE ANALYTICS
# =========================================================
elif menu == "📈 Predictive Analytics":
    st.markdown('<div class="page-title">Predictive Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Pemodelan prediktif dengan Machine Learning</div>', unsafe_allow_html=True)

    if filtered_df.shape[0] < 5:
        st.warning("Data terlalu sedikit untuk analisis prediktif yang stabil.")
    else:
        tab_lr, tab_fc, tab_rf, tab_dt = st.tabs([
            "📉 Regresi Linear", "📈 Forecasting", "🌲 Random Forest", "🌳 Decision Tree"
        ])

        # REGRESI LINEAR
        with tab_lr:
            st.markdown('<div class="section-header">Regresi Linear Interaktif</div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                x_var = st.selectbox("Variabel independen (X)", numeric_cols, index=1 if len(numeric_cols) > 1 else 0, key="lr_x")
            with col2:
                y_var = st.selectbox("Variabel dependen (Y)", numeric_cols, index=0, key="lr_y")

            if x_var == y_var:
                st.warning("Variabel X dan Y tidak boleh sama.")
            else:
                model_df = filtered_df[[x_var, y_var]].dropna().copy()

                if len(model_df) < 3:
                    st.warning("Data tidak cukup untuk regresi linear.")
                else:
                    X = model_df[[x_var]].values
                    y = model_df[y_var].values

                    lr = LinearRegression()
                    lr.fit(X, y)
                    y_pred = lr.predict(X)

                    mae = mean_absolute_error(y, y_pred)
                    rmse = math.sqrt(mean_squared_error(y, y_pred))
                    r2 = r2_score(y, y_pred)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("MAE", f"{mae:.2f}")
                    with col2:
                        st.metric("RMSE", f"{rmse:.2f}")
                    with col3:
                        st.metric("R²", f"{r2:.4f}")

                    st.markdown(
                        f'<div class="insight-card"><b>Persamaan model:</b> {y_var} = {lr.coef_[0]:.4f} × {x_var} + {lr.intercept_:.4f}</div>',
                        unsafe_allow_html=True
                    )

                    plot_df = model_df.copy()
                    plot_df["Prediksi"] = y_pred

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=plot_df[x_var],
                        y=plot_df[y_var],
                        mode="markers",
                        name="Aktual",
                        marker=dict(color="#3b82f6", size=10)
                    ))

                    sort_idx = np.argsort(plot_df[x_var].values)
                    fig.add_trace(go.Scatter(
                        x=plot_df[x_var].values[sort_idx],
                        y=plot_df["Prediksi"].values[sort_idx],
                        mode="lines",
                        name="Garis Regresi",
                        line=dict(color="#10b981", width=3)
                    ))

                    fig.update_layout(
                        title=f"Regresi Linear: {x_var} vs {y_var}",
                        xaxis_title=x_var,
                        yaxis_title=y_var
                    )
                    fig = apply_professional_layout(fig, 540)
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown('<div class="section-header">Prediction Playground</div>', unsafe_allow_html=True)
                    x_input = st.number_input(
                        f"Masukkan nilai {x_var} untuk memprediksi {y_var}",
                        min_value=0.0,
                        value=float(np.median(model_df[x_var]))
                    )
                    pred_val = lr.predict(np.array([[x_input]]))[0]
                    st.markdown(
                        f'<div class="success-card">Prediksi <b>{y_var}</b> untuk nilai <b>{x_var} = {x_input:.2f}</b> adalah <b>{pred_val:.2f}</b> ribu ton.</div>',
                        unsafe_allow_html=True
                    )

        # FORECASTING
        with tab_fc:
            st.markdown('<div class="section-header">Forecasting 2025 (Simulasi Growth Rate)</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="warning-card">Dataset bersifat cross-sectional (1 tahun), sehingga forecasting di dashboard ini diposisikan sebagai simulasi pertumbuhan, bukan time-series forecasting historis.</div>',
                unsafe_allow_html=True
            )

            col1, col2 = st.columns([1, 2])
            with col1:
                commodity_target = st.selectbox("Pilih komoditas untuk forecast", numeric_cols, key="fc_target")
                growth_rate = st.slider("Growth Rate (%)", min_value=1, max_value=20, value=7, key="fc_growth") / 100

            with col2:
                fc_df = filtered_df[["Provinsi", commodity_target]].copy()
                fc_df["Forecast_2025"] = fc_df[commodity_target] * (1 + growth_rate)
                fc_df["Peningkatan"] = fc_df["Forecast_2025"] - fc_df[commodity_target]

                top_fc = fc_df.sort_values(commodity_target, ascending=False).head(min(top_n, len(fc_df))).copy()

                fig = go.Figure()
                fig.add_trace(go.Bar(x=top_fc["Provinsi"], y=top_fc[commodity_target], name="2024", marker_color="#3b82f6"))
                fig.add_trace(go.Bar(x=top_fc["Provinsi"], y=top_fc["Forecast_2025"], name="2025 (Forecast)", marker_color="#10b981"))
                fig.update_layout(
                    barmode="group",
                    title=f"Perbandingan {commodity_target}: 2024 vs 2025",
                    xaxis_title="Provinsi",
                    yaxis_title="Produksi"
                )
                fig = apply_professional_layout(fig, 550)
                st.plotly_chart(fig, use_container_width=True)

            total_now = fc_df[commodity_target].sum()
            total_fc = fc_df["Forecast_2025"].sum()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total 2024", format_num(total_now))
            with col2:
                st.metric("Total 2025 (Forecast)", format_num(total_fc))
            with col3:
                st.metric("Kenaikan Total", format_num(total_fc - total_now))

            st.dataframe(fc_df.sort_values("Forecast_2025", ascending=False), use_container_width=True, hide_index=True)

        # RANDOM FOREST
        with tab_rf:
            st.markdown('<div class="section-header">Random Forest Regression</div>', unsafe_allow_html=True)
            target_rf = st.selectbox("Pilih target prediksi", numeric_cols, key="rf_target")

            feature_cols = [c for c in numeric_cols if c != target_rf]
            rf_df = filtered_df[feature_cols + [target_rf]].dropna().copy()

            if len(rf_df) < 8:
                st.warning("Data terlalu sedikit untuk Random Forest yang stabil.")
            else:
                X = rf_df[feature_cols]
                y = rf_df[target_rf]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )

                rf = RandomForestRegressor(
                    n_estimators=150,
                    max_depth=5,
                    random_state=42
                )
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)

                mae = mean_absolute_error(y_test, y_pred)
                rmse = math.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("MAE", f"{mae:.2f}")
                with col2:
                    st.metric("RMSE", f"{rmse:.2f}")
                with col3:
                    st.metric("R²", f"{r2:.4f}")

                importance = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False).reset_index()
                importance.columns = ["Fitur", "Importance"]

                fig = px.bar(
                    importance,
                    x="Importance",
                    y="Fitur",
                    orientation="h",
                    text="Importance",
                    title=f"Feature Importance Random Forest - {target_rf}"
                )
                fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                fig = apply_professional_layout(fig, 500)
                st.plotly_chart(fig, use_container_width=True)

                compare_df = pd.DataFrame({
                    "Aktual": y_test.values,
                    "Prediksi": y_pred
                })
                st.dataframe(compare_df, use_container_width=True, hide_index=True)

        # DECISION TREE
        with tab_dt:
            st.markdown('<div class="section-header">Decision Tree Regression</div>', unsafe_allow_html=True)
            target_dt = st.selectbox("Pilih target prediksi", numeric_cols, key="dt_target")

            feature_cols_dt = [c for c in numeric_cols if c != target_dt]
            dt_df = filtered_df[feature_cols_dt + [target_dt]].dropna().copy()

            if len(dt_df) < 8:
                st.warning("Data terlalu sedikit untuk Decision Tree yang stabil.")
            else:
                X = dt_df[feature_cols_dt]
                y = dt_df[target_dt]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )

                dt = DecisionTreeRegressor(max_depth=3, random_state=42)
                dt.fit(X_train, y_train)
                y_pred = dt.predict(X_test)

                mae = mean_absolute_error(y_test, y_pred)
                rmse = math.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("MAE", f"{mae:.2f}")
                with col2:
                    st.metric("RMSE", f"{rmse:.2f}")
                with col3:
                    st.metric("R²", f"{r2:.4f}")

                importance = pd.Series(dt.feature_importances_, index=feature_cols_dt).sort_values(ascending=False).reset_index()
                importance.columns = ["Fitur", "Importance"]

                fig = px.bar(
                    importance,
                    x="Importance",
                    y="Fitur",
                    orientation="h",
                    text="Importance",
                    title=f"Feature Importance Decision Tree - {target_dt}"
                )
                fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                fig = apply_professional_layout(fig, 500)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown(
                    '<div class="insight-card">Visualisasi pohon keputusan penuh tidak ditampilkan agar dashboard tetap ringan dan stabil. Interpretasi model difokuskan pada <b>feature importance</b>.</div>',
                    unsafe_allow_html=True
                )

# =========================================================
# INSIGHT & REKOMENDASI
# =========================================================
elif menu == "🧠 Insight & Rekomendasi":
    st.markdown('<div class="page-title">Insight & Rekomendasi</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Temuan analitis dan rekomendasi strategis berbasis data</div>', unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("Tidak ada data untuk filter aktif.")
    else:
        total_by_comm = filtered_df[numeric_cols].sum().sort_values(ascending=False)
        dominant_comm = total_by_comm.index[0]
        dominant_val = total_by_comm.iloc[0]

        top_prov, top_val = top_province_for_commodity(filtered_df, selected_commodity)
        total_selected = filtered_df[selected_commodity].sum()
        share_top = (top_val / total_selected * 100) if total_selected > 0 else 0

        st.markdown('<div class="section-header">💡 Insight Utama</div>', unsafe_allow_html=True)
        
        insights = [
            f"Komoditas dengan total produksi terbesar pada filter aktif adalah <b>{dominant_comm}</b> sebesar <b>{format_num(dominant_val)} ribu ton</b>.",
            f"Untuk komoditas <b>{selected_commodity}</b>, provinsi tertinggi adalah <b>{top_prov}</b> dengan produksi <b>{format_num(top_val)} ribu ton</b>.",
            f"Kontribusi <b>{top_prov}</b> terhadap total <b>{selected_commodity}</b> mencapai sekitar <b>{share_top:.1f}%</b>.",
            "Korelasi antar komoditas perlu dibaca hati-hati karena produksi perkebunan dipengaruhi spesialisasi wilayah, kondisi agroklimat, dan struktur ekonomi regional.",
            "Forecasting pada dashboard ini bersifat simulatif sehingga lebih tepat digunakan untuk eksplorasi skenario awal daripada prediksi kebijakan final."
        ]

        for i, ins in enumerate(insights, start=1):
            st.markdown(f'<div class="insight-card"><b>{i}.</b> {ins}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">🚀 Rekomendasi Strategis</div>', unsafe_allow_html=True)
        
        recs = generate_recommendations(filtered_df, selected_commodity)
        for i, rec in enumerate(recs, start=1):
            st.markdown(f'<div class="success-card"><b>{i}.</b> {rec}</div>', unsafe_allow_html=True)

# =========================================================
# EXPORT CENTER
# =========================================================
elif menu == "📥 Export Center":
    st.markdown('<div class="page-title">Export Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Unduh dataset dan hasil analisis untuk kebutuhan pelaporan</div>', unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("Tidak ada data untuk diekspor.")
    else:
        st.markdown('<div class="section-header">Export Dataset Terfilter</div>', unsafe_allow_html=True)
        
        export_df = add_total_production(filtered_df)
        st.download_button(
            label="⬇️ Download dataset hasil filter (CSV)",
            data=export_csv(export_df),
            file_name="dataset_hasil_filter.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.markdown("---")
        st.markdown('<div class="section-header">Export Forecast 2025</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            exp_comm = st.selectbox("Komoditas untuk forecast", numeric_cols, key="exp_comm")
            exp_growth = st.slider("Growth rate forecast (%)", 1, 20, 7, key="exp_growth") / 100

        with col2:
            export_fc = filtered_df[["Provinsi", exp_comm]].copy()
            export_fc["Forecast_2025"] = export_fc[exp_comm] * (1 + exp_growth)
            export_fc["Peningkatan"] = export_fc["Forecast_2025"] - export_fc[exp_comm]

            st.dataframe(export_fc, use_container_width=True, hide_index=True)

        st.download_button(
            label="⬇️ Download forecast 2025 (CSV)",
            data=export_csv(export_fc),
            file_name=f"forecast_{exp_comm.lower().replace(' ', '_')}_2025.csv",
            mime="text/csv",
            use_container_width=True
        )

# =========================================================
# PROFESSIONAL FOOTER
# =========================================================
st.markdown("""
<div class="professional-footer">
    <h3>🎓 Dashboard UAS Pengenalan Sains Data</h3>
    <p>Versi Professional Dashboard dengan Executive View, Interactive EDA, dan Predictive Analytics</p>
    <p style="margin-top: 1rem;">Dibuat dengan Streamlit, Plotly, dan Scikit-Learn</p>
    <p>© 2026 | Data Source: BPS - Produksi Tanaman Perkebunan 2024</p>
</div>
""", unsafe_allow_html=True)
