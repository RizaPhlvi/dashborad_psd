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

import warnings
warnings.filterwarnings('ignore')

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Dashboard Intelijen Komoditas Perkebunan Indonesia",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# PLANTATION PREMIUM DARK THEME CSS
# =========================================================
st.markdown("""
<style>
    /* =========================================================
       GLOBAL - AGRO-FOREST DARK THEME
    ========================================================= */
    .stApp {
        background: linear-gradient(180deg, #0a1410 0%, #0d1b12 45%, #102015 100%);
        color: #e8f0e4;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    /* =========================================================
       HERO - PLANTATION INTELLIGENCE HEADER
    ========================================================= */
    .hero-wrap {
        background: linear-gradient(135deg, #0d1b12 0%, #1a4d2e 45%, #2f855a 100%);
        padding: 2rem 2.5rem;
        border-radius: 24px;
        color: #ffffff;
        box-shadow: 0 20px 50px rgba(47, 133, 90, 0.3), 0 0 80px rgba(74, 222, 128, 0.08) inset;
        border: 1px solid rgba(186, 230, 170, 0.15);
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }

    .hero-wrap::before {
        content: '🌴';
        position: absolute;
        right: 2rem;
        top: 50%;
        transform: translateY(-50%);
        font-size: 8rem;
        opacity: 0.08;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        line-height: 1.2;
        margin-bottom: 0.5rem;
        color: #ffffff;
        letter-spacing: -0.02em;
        position: relative;
        z-index: 1;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        line-height: 1.6;
        color: rgba(255, 255, 255, 0.9);
        margin-top: 0.35rem;
        max-width: 85%;
        position: relative;
        z-index: 1;
    }

    .hero-badge {
        display: inline-block;
        padding: 0.45rem 1rem;
        border-radius: 999px;
        background: rgba(47, 133, 90, 0.25);
        border: 1px solid rgba(186, 230, 170, 0.3);
        font-size: 0.85rem;
        font-weight: 600;
        color: #a7f3d0;
        margin-right: 0.5rem;
        margin-top: 0.8rem;
        backdrop-filter: blur(10px);
        position: relative;
        z-index: 1;
    }

    /* =========================================================
       SECTION TITLE
    ========================================================= */
    .section-title {
        font-size: 1.3rem;
        font-weight: 800;
        color: #a7f3d0;
        margin-top: 1rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(74, 222, 128, 0.3);
        display: inline-block;
    }

    .section-subtitle {
        font-size: 0.95rem;
        color: #93a39a;
        margin-bottom: 1.5rem;
        font-style: italic;
    }

    .subtle-text {
        color: #c7d2c0;
        font-size: 0.96rem;
        line-height: 1.6;
    }

    /* =========================================================
       PLANTATION INFO BOXES
    ========================================================= */
    .plantation-card {
        background: linear-gradient(180deg, #13261a 0%, #0d1b12 100%);
        border: 1px solid rgba(186, 230, 170, 0.1);
        border-radius: 18px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 10px 26px rgba(0, 0, 0, 0.35);
    }

    .info-box {
        background: linear-gradient(135deg, rgba(47, 133, 90, 0.18) 0%, rgba(74, 222, 128, 0.08) 100%);
        border-left: 4px solid #4ade80;
        padding: 1.1rem 1.25rem;
        border-radius: 14px;
        color: #d1fae5;
        margin: 0.5rem 0 1rem 0;
        box-shadow: 0 6px 18px rgba(47, 133, 90, 0.15);
        line-height: 1.65;
    }

    .success-box {
        background: linear-gradient(135deg, rgba(132, 204, 22, 0.16) 0%, rgba(163, 230, 53, 0.08) 100%);
        border-left: 4px solid #84cc16;
        padding: 1.1rem 1.25rem;
        border-radius: 14px;
        color: #ecfccb;
        margin: 0.5rem 0 1rem 0;
        box-shadow: 0 6px 18px rgba(132, 204, 22, 0.12);
        line-height: 1.65;
    }

    .warn-box {
        background: linear-gradient(135deg, rgba(212, 160, 23, 0.18) 0%, rgba(251, 191, 36, 0.08) 100%);
        border-left: 4px solid #d4a017;
        padding: 1.1rem 1.25rem;
        border-radius: 14px;
        color: #fef3c7;
        margin: 0.5rem 0 1rem 0;
        box-shadow: 0 6px 18px rgba(212, 160, 23, 0.12);
        line-height: 1.65;
    }

    .insight-box {
        background: linear-gradient(135deg, rgba(124, 79, 42, 0.15) 0%, rgba(139, 94, 52, 0.08) 100%);
        border-left: 4px solid #c7815e;
        padding: 1.1rem 1.25rem;
        border-radius: 14px;
        color: #fef3c7;
        margin: 0.5rem 0 1rem 0;
        box-shadow: 0 6px 18px rgba(124, 79, 42, 0.12);
        line-height: 1.65;
    }

    .footer-box {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #0d1b12 0%, #1a4d2e 100%);
        color: #a7f3d0;
        border-radius: 18px;
        margin-top: 1.5rem;
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(186, 230, 170, 0.12);
    }

    /* =========================================================
       SIDEBAR - PLANTATION CONTROL CENTER
    ========================================================= */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1410 0%, #0d1b12 100%);
        border-right: 1px solid rgba(186, 230, 170, 0.1);
    }

    section[data-testid="stSidebar"] * {
        color: #e8f0e4 !important;
    }

    section[data-testid="stSidebar"] h2 {
        color: #4ade80 !important;
        font-weight: 700 !important;
    }

    /* =========================================================
       METRIC CARDS - PLANTATION STYLE
    ========================================================= */
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, #13261a 0%, #0d1b12 100%);
        border: 1px solid rgba(186, 230, 170, 0.1);
        border-radius: 18px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }

    div[data-testid="stMetric"]:hover {
        border-color: rgba(74, 222, 128, 0.4);
        box-shadow: 0 12px 28px rgba(74, 222, 128, 0.15);
    }

    div[data-testid="stMetricLabel"] {
        color: #93a39a !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    div[data-testid="stMetricValue"] {
        color: #a7f3d0 !important;
        font-weight: 800 !important;
        font-size: 1.9rem !important;
    }

    div[data-testid="stMetricDelta"] {
        color: #4ade80 !important;
    }

    /* =========================================================
       TABS - PLANTATION NAVIGATION
    ========================================================= */
    button[data-baseweb="tab"] {
        border-radius: 10px !important;
        font-weight: 700 !important;
        background: #13261a !important;
        color: #c7d2c0 !important;
        border: 1px solid rgba(186, 230, 170, 0.1) !important;
        padding: 0.75rem 1.25rem !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #1a4d2e 0%, #2f855a 100%) !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(47, 133, 90, 0.4) !important;
    }

    /* =========================================================
       DATAFRAME / TABLE
    ========================================================= */
    .stDataFrame, .stTable {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(186, 230, 170, 0.1) !important;
    }

    div[data-testid="stExpander"] {
        background: #13261a;
        border: 1px solid rgba(186, 230, 170, 0.1);
        border-radius: 14px;
    }

    /* =========================================================
       INPUTS
    ========================================================= */
    .stSelectbox label,
    .stSlider label,
    .stRadio label,
    .stMultiSelect label,
    .stNumberInput label {
        color: #c7d2c0 !important;
        font-weight: 600;
    }

    /* =========================================================
       PLANTATION COMMODITY PILLS
    ========================================================= */
    .commodity-pill {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
        color: white;
    }

    header[data-testid="stHeader"] {
        background: rgba(0, 0, 0, 0);
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# DATASET EMBEDDED - DATA PERKEBUNAN INDONESIA 2024
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
PAPUA PEGUNUNGAN,0,0.02,0,3.27,0,0,0"""

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():
    return pd.read_csv(io.StringIO(CSV_DATA))

df = load_data()
numeric_cols = [c for c in df.columns if c != "Provinsi"]

# =========================================================
# PLANTATION COMMODITY IDENTITY MAP
# =========================================================
COMMODITY_IDENTITY = {
    "Kelapa Sawit": {
        "icon": "🌴",
        "color": "#2f855a",
        "color_light": "#4ade80",
        "character": "Komoditas primadona ekspor Indonesia. Mendominasi perkebunan Sumatera dan Kalimantan sebagai sentra CPO nasional.",
        "sector": "Perkebunan Besar"
    },
    "Kelapa": {
        "icon": "🥥",
        "color": "#38bdf8",
        "color_light": "#7dd3fc",
        "character": "Tanaman rakyat yang tersebar luas di pesisir Nusantara. Potensi hilirisasi minyak kelapa dan turunan sabut sangat tinggi.",
        "sector": "Perkebunan Rakyat"
    },
    "Karet": {
        "icon": "🌳",
        "color": "#7c4f2a",
        "color_light": "#c7815e",
        "character": "Komoditas strategis dengan sentra utama di Sumatera. Menghadapi tantangan fluktuasi harga global dan peremajaan lahan.",
        "sector": "Perkebunan Campuran"
    },
    "Kopi": {
        "icon": "☕",
        "color": "#8b5e34",
        "color_light": "#d4a574",
        "character": "Identitas budaya Indonesia. Kopi Gayo, Toraja, Kintamani, dan Java Preanger menjadi ikon specialty coffee dunia.",
        "sector": "Perkebunan Rakyat"
    },
    "Kakao": {
        "icon": "🍫",
        "color": "#6b3410",
        "color_light": "#a0522d",
        "character": "Sulawesi sebagai tulang punggung produksi kakao nasional. Perlu peremajaan pohon dan peningkatan fermentasi pasca panen.",
        "sector": "Perkebunan Rakyat"
    },
    "Teh": {
        "icon": "🍃",
        "color": "#059669",
        "color_light": "#34d399",
        "character": "Eksklusif tumbuh di dataran tinggi berhawa dingin. Jawa Barat dan Sumatera Utara menjadi sentra utama perkebunan teh.",
        "sector": "Perkebunan Besar"
    },
    "Tebu": {
        "icon": "🌾",
        "color": "#84cc16",
        "color_light": "#bef264",
        "character": "Bahan baku gula nasional yang terkonsentrasi di Jawa Timur dan Lampung. Vital untuk ketahanan pangan dan gula domestik.",
        "sector": "Perkebunan Strategis"
    }
}

def get_commodity_color(commodity, light=False):
    """Ambil warna identitas komoditas"""
    if commodity in COMMODITY_IDENTITY:
        return COMMODITY_IDENTITY[commodity]["color_light"] if light else COMMODITY_IDENTITY[commodity]["color"]
    return "#4ade80"

def get_commodity_icon(commodity):
    """Ambil icon komoditas"""
    return COMMODITY_IDENTITY.get(commodity, {}).get("icon", "🌱")

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def format_num(x):
    """Format angka ke K/M suffix"""
    try:
        val = float(x)
        if val >= 1_000_000:
            return f"{val/1_000_000:.1f}M"
        if val >= 1_000:
            return f"{val/1_000:.1f}K"
        return f"{val:,.0f}"
    except:
        return str(x)

def format_ton(x):
    """Format ribu ton"""
    try:
        return f"{float(x):,.2f}"
    except:
        return str(x)

def get_filtered_df(base_df, province_filter, show_zero_rows):
    """Filter data perkebunan"""
    data = base_df.copy()
    if province_filter != "Semua Provinsi":
        data = data[data["Provinsi"] == province_filter].copy()
    if not show_zero_rows:
        data = data[(data[numeric_cols].sum(axis=1) > 0)].copy()
    return data

def add_total_production(data):
    """Tambah kolom total produksi perkebunan"""
    temp = data.copy()
    temp["Total Produksi"] = temp[numeric_cols].sum(axis=1)
    return temp

def top_province_for_commodity(data, commodity):
    """Cari provinsi sentra produksi komoditas"""
    if data.empty or commodity not in data.columns:
        return "-", 0
    idx = data[commodity].idxmax()
    return data.loc[idx, "Provinsi"], data.loc[idx, commodity]

def province_with_highest_total(data):
    """Cari provinsi dengan total produksi perkebunan tertinggi"""
    temp = add_total_production(data)
    if temp.empty:
        return "-", 0
    idx = temp["Total Produksi"].idxmax()
    return temp.loc[idx, "Provinsi"], temp.loc[idx, "Total Produksi"]

def province_with_most_diverse(data):
    """Cari provinsi dengan portofolio komoditas paling terdiversifikasi"""
    if data.empty:
        return "-", 0
    diversity = (data[numeric_cols] > 0).sum(axis=1)
    idx = diversity.idxmax()
    return data.loc[idx, "Provinsi"], diversity[idx]

def generate_dynamic_insight(data, commodity, context="general"):
    """Generate insight kontekstual perkebunan"""
    if data.empty:
        return "Tidak ada data perkebunan yang tersedia untuk filter saat ini."

    total = data[commodity].sum()
    if total <= 0:
        return f"Komoditas <b>{commodity}</b> belum tercatat memiliki sentra produksi pada wilayah filter aktif. Potensi pengembangan masih terbuka."

    icon = get_commodity_icon(commodity)
    top_df = data.nlargest(3, commodity)[["Provinsi", commodity]].copy()
    top_df["Share"] = (top_df[commodity] / total) * 100
    top1 = top_df.iloc[0]
    active_provs = (data[commodity] > 0).sum()

    if context == "commodity":
        msg = (
            f"{icon} <b>Profil Produksi {commodity}</b><br>"
            f"Total produksi nasional pada filter aktif mencapai <b>{format_ton(total)} ribu ton</b>. "
            f"<b>{top1['Provinsi']}</b> berperan sebagai <b>sentra utama</b> dengan kontribusi <b>{format_ton(top1[commodity])} ribu ton</b> "
            f"(<b>{top1['Share']:.1f}%</b> dari total). "
        )
    elif context == "national":
        msg = (
            f"{icon} <b>{commodity}</b> diproduksi oleh <b>{active_provs} dari {len(data)}</b> provinsi aktif, "
            f"menunjukkan pola <b>spesialisasi geografis</b> yang khas pada sektor perkebunan. "
            f"Sentra utama <b>{top1['Provinsi']}</b> menguasai <b>{top1['Share']:.1f}%</b> produksi nasional."
        )
    else:
        msg = (
            f"{icon} Produksi <b>{commodity}</b> pada filter aktif mencapai <b>{format_ton(total)} ribu ton</b>. "
            f"Kontributor terbesar adalah <b>{top1['Provinsi']}</b> sebagai sentra produksi dengan porsi "
            f"<b>{top1['Share']:.1f}%</b>."
        )

    if len(top_df) >= 3:
        share3 = top_df["Share"].sum()
        if share3 > 70:
            msg += f" <br><b>⚠️ Konsentrasi tinggi:</b> tiga provinsi teratas menyumbang <b>{share3:.1f}%</b> — menunjukkan dominasi sentra produksi berskala besar."
        elif share3 > 50:
            msg += f" <br><b>📍 Pola sentra:</b> tiga provinsi teratas berkontribusi <b>{share3:.1f}%</b>, menunjukkan basis geografis yang terkonsentrasi."
        else:
            msg += f" <br><b>🌱 Distribusi merata:</b> tiga provinsi teratas hanya menyumbang <b>{share3:.1f}%</b>, menunjukkan sebaran produksi yang relatif luas."

    return msg

def generate_province_insight(data, province):
    """Generate insight khusus profil provinsi perkebunan"""
    if data.empty:
        return "Data provinsi tidak tersedia."

    p_row = data[data["Provinsi"] == province].iloc[0]
    prod_data = {c: p_row[c] for c in numeric_cols}
    total = sum(prod_data.values())

    if total == 0:
        return f"🏙️ <b>{province}</b> tidak memiliki basis produksi perkebunan signifikan, cenderung merupakan wilayah urban."

    active = {k: v for k, v in prod_data.items() if v > 0}
    dominant = max(active.items(), key=lambda x: x[1])
    diversity = len(active)

    dom_icon = get_commodity_icon(dominant[0])
    dom_share = (dominant[1] / total) * 100

    msg = f"{dom_icon} <b>{province}</b> memiliki <b>total produksi perkebunan {format_ton(total)} ribu ton</b>. "

    if diversity >= 5:
        msg += f"Provinsi ini menunjukkan <b>portofolio komoditas terdiversifikasi</b> dengan <b>{diversity} komoditas aktif</b>. "
    elif diversity >= 3:
        msg += f"Terdapat <b>{diversity} komoditas aktif</b> yang menopang sektor perkebunan daerah ini. "
    else:
        msg += f"Produksi sangat <b>terspesialisasi</b> pada <b>{diversity} komoditas</b>. "

    msg += f"<b>{dominant[0]}</b> berperan sebagai <b>komoditas andalan</b> dengan kontribusi <b>{dom_share:.1f}%</b> dari total produksi provinsi."

    if dom_share > 70:
        msg += " <br><b>⚠️ Ketergantungan tinggi:</b> struktur produksi sangat bergantung pada satu komoditas — rentan terhadap guncangan harga pasar."
    elif dom_share > 50:
        msg += " <br><b>📊 Basis kuat:</b> komoditas dominan menjadi penopang utama, namun diversifikasi dapat memperkuat ketahanan ekonomi daerah."

    return msg

def generate_recommendations(data, commodity):
    """Generate rekomendasi strategis perkebunan"""
    if data.empty:
        return ["Tidak ada data yang tersedia untuk menghasilkan rekomendasi."]

    total_by_commodity = data[numeric_cols].sum().sort_values(ascending=False)
    dominant = total_by_commodity.index[0]
    dominant_val = total_by_commodity.iloc[0]
    dom_icon = get_commodity_icon(dominant)

    top_prov, top_val = top_province_for_commodity(data, commodity)
    total_selected = data[commodity].sum()
    share = (top_val / total_selected * 100) if total_selected > 0 else 0
    comm_icon = get_commodity_icon(commodity)

    recs = [
        f"{dom_icon} <b>Strategi Hilirisasi {dominant}:</b> Prioritaskan pengembangan pabrik pengolahan di sentra produksi untuk meningkatkan nilai tambah ekspor, bukan sekadar mengekspor bahan mentah ({format_ton(dominant_val)} ribu ton).",

        f"{comm_icon} <b>Mitigasi Konsentrasi Wilayah {commodity}:</b> Dengan <b>{top_prov}</b> menyumbang <b>{share:.1f}%</b> produksi, perlu pengembangan sentra alternatif di provinsi potensial untuk mengurangi risiko iklim dan pasar.",

        "🌳 <b>Program Peremajaan Lahan:</b> Banyak tanaman perkebunan Indonesia telah berusia tua (>25 tahun). Replanting dengan bibit unggul akan meningkatkan produktivitas 40-60% dalam 5 tahun ke depan.",

        "🗺️ <b>Penguatan Klaster Komoditas:</b> Bentuk klaster perkebunan tematik per wilayah (klaster sawit Sumatera, kakao Sulawesi, kopi Jawa) untuk efisiensi supply chain dan penguatan branding.",

        "📊 <b>Monitoring Data Time-Series:</b> Kembangkan sistem pencatatan data perkebunan berbasis time-series untuk forecasting produksi yang lebih akurat dan perencanaan kebijakan strategis."
    ]
    return recs

def export_csv(dataframe):
    return dataframe.to_csv(index=False).encode("utf-8")

# =========================================================
# PLOTLY PLANTATION THEME
# =========================================================
plot_bg = "#0d1b12"
paper_bg = "#0d1b12"
font_color = "#e8f0e4"
grid_color = "rgba(186, 230, 170, 0.08)"

PLANTATION_PALETTE = [
    "#2f855a",  # hijau sawit
    "#8b5e34",  # kopi
    "#6b3410",  # kakao
    "#7c4f2a",  # karet
    "#38bdf8",  # kelapa
    "#84cc16",  # tebu
    "#059669",  # teh
    "#d4a017",  # emas panen
    "#c7815e",  # amber
    "#a3e635"   # lime
]

def apply_plantation_layout(fig, height=500):
    """Apply plantation theme ke chart Plotly"""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(color=font_color, family="Inter, sans-serif"),
        height=height,
        margin=dict(l=40, r=30, t=60, b=40),
        xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(186, 230, 170, 0.1)"
        )
    )
    return fig

# =========================================================
# HERO HEADER - PLANTATION INTELLIGENCE
# =========================================================
st.markdown("""
<div class="hero-wrap">
    <div class="hero-title">🌴 Dashboard Intelijen Komoditas Perkebunan Indonesia</div>
    <div class="hero-subtitle">
        Platform analitik strategis untuk pemetaan sentra produksi, pemetaan komoditas andalan daerah, 
        dan eksplorasi pola produksi perkebunan nasional per provinsi. 
        Dirancang sebagai pusat intelijen sektor perkebunan — mendukung perencanaan kebijakan, 
        investasi hilirisasi, dan pengembangan agro-industri Indonesia.
    </div>
    <div>
        <span class="hero-badge">🌾 Sentra Produksi</span>
        <span class="hero-badge">📊 Portofolio Komoditas</span>
        <span class="hero-badge">🗺️ Intelijen Wilayah</span>
        <span class="hero-badge">📈 Proyeksi Panen</span>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR - PLANTATION CONTROL CENTER
# =========================================================
st.sidebar.title("🎛️ Pusat Kendali Perkebunan")

menu = st.sidebar.radio(
    "🧭 Navigasi Intelijen",
    [
        "🏠 Ringkasan Perkebunan Nasional",
        "🌴 Profil Komoditas",
        "🗺️ Profil Provinsi Perkebunan",
        "📊 Analisis Produksi & Korelasi",
        "🌍 Sebaran Wilayah Produksi",
        "📈 Proyeksi & Model Produksi",
        "🧠 Insight & Strategi Perkebunan",
        "📦 Data & Ekspor"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🌱 Filter Komoditas & Wilayah")

selected_commodity = st.sidebar.selectbox(
    "🌾 Komoditas Fokus",
    numeric_cols,
    index=0,
    help="Pilih komoditas perkebunan untuk dianalisis mendalam"
)

# Tampilkan identitas komoditas
comm_info = COMMODITY_IDENTITY[selected_commodity]
st.sidebar.markdown(f"""
<div style="padding: 0.8rem; background: rgba(47, 133, 90, 0.15); border-radius: 10px; border-left: 3px solid {comm_info['color']}; margin-top: 0.5rem;">
    <div style="font-size: 0.75rem; color: #93a39a; text-transform: uppercase;">Sektor</div>
    <div style="font-size: 0.9rem; color: #a7f3d0; font-weight: 600;">{comm_info['sector']}</div>
    <div style="font-size: 0.75rem; color: #93a39a; margin-top: 0.3rem;">{comm_info['character'][:80]}...</div>
</div>
""", unsafe_allow_html=True)

province_options = ["Semua Provinsi"] + df["Provinsi"].tolist()
selected_province = st.sidebar.selectbox("🗺️ Wilayah Provinsi", province_options, index=0)

top_n = st.sidebar.slider("🏆 Top N Sentra Produksi", min_value=5, max_value=20, value=10)
show_zero_rows = st.sidebar.checkbox("Tampilkan wilayah tanpa produksi", value=True)

view_mode = st.sidebar.radio(
    "📊 Mode Tampilan",
    ["Volume Produksi (Absolut)", "Pangsa (%)"],
    index=0
)

filtered_df = get_filtered_df(df, selected_province, show_zero_rows)

# Ringkasan data aktif di sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="padding: 0.8rem; background: rgba(74, 222, 128, 0.08); border-radius: 10px; border: 1px solid rgba(186, 230, 170, 0.15); font-size: 0.85rem;">
    <div style="color: #4ade80; font-weight: 700; margin-bottom: 0.3rem;">📡 Status Data Aktif</div>
    <div style="color: #c7d2c0;">Wilayah aktif: <b>{}</b></div>
    <div style="color: #c7d2c0;">Total observasi: <b>{}</b></div>
</div>
""".format(selected_province, len(filtered_df)), unsafe_allow_html=True)

# =========================================================
# PAGE 1: RINGKASAN PERKEBUNAN NASIONAL
# =========================================================
if menu == "🏠 Ringkasan Perkebunan Nasional":
    st.markdown('<div class="section-title">🏠 Ringkasan Perkebunan Nasional</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Potret menyeluruh sektor perkebunan Indonesia — sentra produksi, komoditas dominan, dan struktur wilayah</div>', unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("Tidak ada data perkebunan yang cocok dengan filter aktif.")
    else:
        total_production = filtered_df[numeric_cols].sum().sum()
        total_province = filtered_df.shape[0]
        commodity_totals = filtered_df[numeric_cols].sum().sort_values(ascending=False)
        top_commodity = commodity_totals.index[0]
        top_commodity_val = commodity_totals.iloc[0]
        top_commodity_icon = get_commodity_icon(top_commodity)
        best_prov, best_total = province_with_highest_total(filtered_df)
        diverse_prov, diversity_score = province_with_most_diverse(filtered_df)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🌾 Total Produksi Nasional", format_num(total_production), "Ribu Ton")
        c2.metric(f"{top_commodity_icon} Komoditas Dominan", top_commodity, format_ton(top_commodity_val))
        c3.metric("🏆 Sentra Produksi Utama", best_prov[:14], format_ton(best_total))
        c4.metric("🌱 Provinsi Terdiversifikasi", diverse_prov[:14], f"{int(diversity_score)} komoditas")

        st.markdown("")

        left, right = st.columns([1.2, 1])

        with left:
            st.markdown(f'<div class="section-title">🌾 Top {top_n} Sentra Produksi — {selected_commodity}</div>', unsafe_allow_html=True)
            top_df = filtered_df[["Provinsi", selected_commodity]].sort_values(selected_commodity, ascending=False).head(min(top_n, len(filtered_df))).copy()

            if view_mode == "Pangsa (%)":
                total_val = top_df[selected_commodity].sum()
                top_df["Display"] = (top_df[selected_commodity] / total_val * 100) if total_val > 0 else 0
                x_col = "Display"
                x_title = "Pangsa (%)"
            else:
                x_col = selected_commodity
                x_title = "Produksi (Ribu Ton)"

            comm_color = get_commodity_color(selected_commodity, light=True)
            fig = px.bar(
                top_df.sort_values(x_col, ascending=True),
                x=x_col,
                y="Provinsi",
                orientation="h",
                text=x_col,
                title=f"Sentra Produksi {selected_commodity}",
                color_discrete_sequence=[comm_color]
            )
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_layout(xaxis_title=x_title, yaxis_title="")
            fig = apply_plantation_layout(fig, 540)
            st.plotly_chart(fig, use_container_width=True, key="exec_bar_chart_plantation")

        with right:
            st.markdown('<div class="section-title">🥧 Komposisi Produksi Perkebunan</div>', unsafe_allow_html=True)
            comp_df = filtered_df[numeric_cols].sum().reset_index()
            comp_df.columns = ["Komoditas", "Produksi"]
            if view_mode == "Pangsa (%)":
                total_comp = comp_df["Produksi"].sum()
                comp_df["Produksi"] = (comp_df["Produksi"] / total_comp * 100) if total_comp > 0 else 0

            comm_colors = [get_commodity_color(c, light=True) for c in comp_df["Komoditas"]]
            fig2 = px.pie(
                comp_df,
                values="Produksi",
                names="Komoditas",
                hole=0.55,
                color="Komoditas",
                color_discrete_map={c: get_commodity_color(c, light=True) for c in comp_df["Komoditas"]}
            )
            fig2.update_traces(textinfo='percent+label')
            fig2 = apply_plantation_layout(fig2, 540)
            fig2.update_layout(showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.2))
            st.plotly_chart(fig2, use_container_width=True, key="exec_pie_chart_plantation")

        st.markdown('<div class="section-title">💡 Intelijen Singkat</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="info-box">{generate_dynamic_insight(filtered_df, selected_commodity, "national")}</div>',
            unsafe_allow_html=True
        )

        st.markdown('<div class="section-title">📋 Peta Produksi per Wilayah</div>', unsafe_allow_html=True)
        summary_df = add_total_production(filtered_df)
        st.dataframe(summary_df, use_container_width=True)

# =========================================================
# PAGE 2: PROFIL KOMODITAS
# =========================================================
elif menu == "🌴 Profil Komoditas":
    target_comm = st.selectbox(
        "🌾 Pilih Komoditas Perkebunan untuk Dianalisis",
        numeric_cols,
        key="comm_int_sel"
    )

    comm_info = COMMODITY_IDENTITY[target_comm]
    st.markdown(f'<div class="section-title">{comm_info["icon"]} Profil Komoditas: {target_comm}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-subtitle">Sektor: <b>{comm_info["sector"]}</b> — {comm_info["character"]}</div>', unsafe_allow_html=True)

    comm_df = filtered_df[["Provinsi", target_comm]].copy()
    comm_df = comm_df[comm_df[target_comm] > 0].sort_values(target_comm, ascending=False)

    tot_c = comm_df[target_comm].sum()
    top_prov_c = comm_df.iloc[0]["Provinsi"] if not comm_df.empty else "-"
    top_val_c = comm_df.iloc[0][target_comm] if not comm_df.empty else 0
    med_c = comm_df[target_comm].median() if not comm_df.empty else 0
    top5_c_share = comm_df.head(5)[target_comm].sum() / max(1, tot_c) * 100
    active_provs = len(comm_df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 Total Panen Nasional", format_num(tot_c), "Ribu Ton")
    c2.metric("🏆 Sentra Utama", top_prov_c[:14], format_ton(top_val_c))
    c3.metric("⚖️ Median Produksi", format_ton(med_c if not pd.isna(med_c) else 0), "Ribu Ton")
    c4.metric("🌐 Provinsi Aktif", f"{active_provs}", f"Share Top-5: {top5_c_share:.1f}%")

    tab1, tab2, tab3 = st.tabs(["🏆 Hirarki Sentra Produksi", "📊 Potret Distribusi", "🗺️ Kontribusi Wilayah"])

    comm_color = get_commodity_color(target_comm, light=True)

    with tab1:
        fig = px.bar(
            comm_df.head(top_n)[::-1],
            x=target_comm,
            y="Provinsi",
            orientation='h',
            text=target_comm,
            title=f"Hirarki Sentra Produksi {target_comm}",
            color_discrete_sequence=[comm_color]
        )
        fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        fig = apply_plantation_layout(fig, 600)
        fig.update_layout(xaxis_title="Produksi (Ribu Ton)")
        st.plotly_chart(fig, use_container_width=True, key=f"comm_rank_{target_comm}")

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            fig_h = px.histogram(
                comm_df,
                x=target_comm,
                nbins=15,
                color_discrete_sequence=[comm_color],
                title="Distribusi Volume Produksi"
            )
            fig_h = apply_plantation_layout(fig_h, 450)
            st.plotly_chart(fig_h, use_container_width=True, key=f"comm_hist_{target_comm}")
        with c2:
            fig_b = px.box(
                comm_df,
                y=target_comm,
                color_discrete_sequence=[comm_color],
                points="all",
                title="Pola Persebaran & Deteksi Sentra Ekstrem"
            )
            fig_b = apply_plantation_layout(fig_b, 450)
            st.plotly_chart(fig_b, use_container_width=True, key=f"comm_box_{target_comm}")

    with tab3:
        fig_t = px.treemap(
            comm_df.head(15),
            path=["Provinsi"],
            values=target_comm,
            color=target_comm,
            color_continuous_scale=[comm_info["color"], comm_info["color_light"]],
            title="Peta Kontribusi Wilayah terhadap Produksi Nasional"
        )
        fig_t = apply_plantation_layout(fig_t, 600)
        st.plotly_chart(fig_t, use_container_width=True, key=f"comm_tree_{target_comm}")

    st.markdown(f'<div class="insight-box">{generate_dynamic_insight(filtered_df, target_comm, "commodity")}</div>', unsafe_allow_html=True)

# =========================================================
# PAGE 3: PROFIL PROVINSI PERKEBUNAN
# =========================================================
elif menu == "🗺️ Profil Provinsi Perkebunan":
    target_prov = st.selectbox(
        "🗺️ Pilih Provinsi untuk Profil Perkebunan",
        df["Provinsi"].tolist(),
        key="prov_int_sel"
    )

    p_df = df[df["Provinsi"] == target_prov].iloc[0]

    p_profile = pd.DataFrame({
        "Komoditas": numeric_cols,
        "Produksi": [p_df[c] for c in numeric_cols]
    }).sort_values("Produksi", ascending=False)

    tot_p = p_profile["Produksi"].sum()
    dom_p = p_profile.iloc[0]["Kommodity"] if "Kommodity" in p_profile.columns else p_profile.iloc[0]["Komoditas"]
    dom_p = p_profile.iloc[0]["Komoditas"]
    active_c = (p_profile["Produksi"] > 0).sum()

    # National average comparison
    nat_avg = pd.DataFrame({
        "Komoditas": numeric_cols,
        "Rata_Nasional": [df[c].mean() for c in numeric_cols]
    })
    comp_df = p_profile.merge(nat_avg, on="Komoditas", suffixes=("_Prov", "_Nat"))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🌾 Total Panen Provinsi", format_num(tot_p), "Ribu Ton")
    c2.metric(f"{get_commodity_icon(dom_p)} Komoditas Andalan", dom_p)
    c3.metric("🌱 Sektor Aktif", f"{int(active_c)}", f"dari {len(numeric_cols)} komoditas")
    dom_share = (p_profile.iloc[0]['Produksi'] / max(1, tot_p) * 100) if tot_p > 0 else 0
    c4.metric("📊 Konsentrasi Produksi", f"{dom_share:.0f}%", "Share komoditas andalan")

    st.markdown(f'<div class="section-title">🌳 Potret Komoditas Provinsi {target_prov}</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        # Radar Chart - Portofolio Komoditas
        fig_r = go.Figure()

        # Normalisasi untuk radar (skala relatif terhadap max)
        p_values = p_profile["Produksi"].tolist()
        p_max = max(p_values) if p_values else 1
        p_norm = [(v / p_max * 100) if p_max > 0 else 0 for v in p_values]

        nat_values = nat_avg["Rata_Nasional"].tolist()
        nat_max = max(nat_values) if nat_values else 1
        nat_norm = [(v / nat_max * 100) if nat_max > 0 else 0 for v in nat_values]

        fig_r.add_trace(go.Scatterpolar(
            r=p_norm + [p_norm[0]],
            theta=p_profile["Komoditas"].tolist() + [p_profile["Komoditas"].iloc[0]],
            fill='toself',
            name=target_prov,
            line_color="#4ade80",
            fillcolor="rgba(74, 222, 128, 0.3)"
        ))
        fig_r.add_trace(go.Scatterpolar(
            r=nat_norm + [nat_norm[0]],
            theta=nat_avg["Komoditas"].tolist() + [nat_avg["Komoditas"].iloc[0]],
            fill='toself',
            name="Rata-rata Nasional",
            line_color="#d4a017",
            fillcolor="rgba(212, 160, 23, 0.15)"
        ))
        fig_r.update_layout(
            polar=dict(
                bgcolor="rgba(13, 27, 18, 0.5)",
                radialaxis=dict(visible=True, showticklabels=False, gridcolor="rgba(186, 230, 170, 0.15)"),
                angularaxis=dict(gridcolor="rgba(186, 230, 170, 0.15)")
            ),
            showlegend=True,
            title=f"Portofolio Komoditas {target_prov} vs Nasional"
        )
        fig_r = apply_plantation_layout(fig_r, 550)
        st.plotly_chart(fig_r, use_container_width=True, key=f"prov_radar_{target_prov}")

    with col2:
        # Bar chart produksi per komoditas
        color_map = {row["Komoditas"]: get_commodity_color(row["Komoditas"], light=True)
                    for _, row in p_profile.iterrows()}

        fig_bar = px.bar(
            p_profile,
            x="Komoditas",
            y="Produksi",
            text="Produksi",
            color="Komoditas",
            color_discrete_map=color_map,
            title="Volume Produksi per Komoditas"
        )
        fig_bar.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        fig_bar.update_layout(showlegend=False)
        fig_bar = apply_plantation_layout(fig_bar, 550)
        st.plotly_chart(fig_bar, use_container_width=True, key=f"prov_bar_{target_prov}")

    # Benchmark diverging
    st.markdown('<div class="section-title">⚖️ Posisi Relatif terhadap Rata-rata Nasional</div>', unsafe_allow_html=True)
    fig_div = go.Figure()
    deviation = comp_df["Produksi"] - comp_df["Rata_Nasional"]
    colors = ["#4ade80" if v >= 0 else "#c7815e" for v in deviation]
    fig_div.add_trace(go.Bar(
        y=comp_df["Komoditas"],
        x=deviation,
        orientation='h',
        marker_color=colors,
        name="Deviasi"
    ))
    fig_div.update_layout(
        title=f"Posisi {target_prov} vs Rata-rata Nasional (Hijau = Di Atas, Cokelat = Di Bawah)",
        xaxis_title="Deviasi (Ribu Ton)"
    )
    fig_div = apply_plantation_layout(fig_div, 450)
    st.plotly_chart(fig_div, use_container_width=True, key=f"prov_div_{target_prov}")

    st.markdown(f'<div class="insight-box">{generate_province_insight(df, target_prov)}</div>', unsafe_allow_html=True)

# =========================================================
# PAGE 4: ANALISIS PRODUKSI & KORELASI
# =========================================================
elif menu == "📊 Analisis Produksi & Korelasi":
    st.markdown('<div class="section-title">📊 Analisis Produksi & Korelasi</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Eksplorasi pola produksi, hubungan antar-komoditas, dan struktur korelasi sektor perkebunan</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "⚔️ Duel Dua Wilayah",
        "🔗 Relasi Antar-Komoditas",
        "🌡️ Matriks Korelasi",
        "🎯 Benchmarking Wilayah"
    ])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            p1 = st.selectbox("🏆 Wilayah A", df["Provinsi"].tolist(), index=0, key="comp_prov_a")
        with c2:
            p2 = st.selectbox("🏆 Wilayah B", df["Provinsi"].tolist(), index=3, key="comp_prov_b")

        p1_data = df[df["Provinsi"] == p1].iloc[0]
        p2_data = df[df["Provinsi"] == p2].iloc[0]

        compare_df = pd.DataFrame({
            "Komoditas": numeric_cols,
            p1: [p1_data[c] for c in numeric_cols],
            p2: [p2_data[c] for c in numeric_cols]
        })

        fig = go.Figure()
        fig.add_trace(go.Bar(name=p1, x=compare_df["Komoditas"], y=compare_df[p1], marker_color="#4ade80"))
        fig.add_trace(go.Bar(name=p2, x=compare_df["Komoditas"], y=compare_df[p2], marker_color="#d4a017"))
        fig.update_layout(barmode='group', title=f"Perbandingan Portofolio Produksi: {p1} vs {p2}")
        fig = apply_plantation_layout(fig, 500)
        st.plotly_chart(fig, use_container_width=True, key="comp_prov_vs_prov")

        p1_wins = (compare_df[p1] > compare_df[p2]).sum()
        p2_wins = (compare_df[p2] > compare_df[p1]).sum()
        p1_total = compare_df[p1].sum()
        p2_total = compare_df[p2].sum()
        leader = p1 if p1_total > p2_total else p2

        st.markdown(f"""
        <div class="info-box">
        <b>⚔️ Hasil Perbandingan:</b><br>
        • <b>{p1}</b> unggul di <b>{p1_wins}</b> komoditas | <b>{p2}</b> unggul di <b>{p2_wins}</b> komoditas<br>
        • Total produksi: <b>{p1}</b> = {format_ton(p1_total)} | <b>{p2}</b> = {format_ton(p2_total)} ribu ton<br>
        • <b>{leader}</b> menjadi wilayah dengan portofolio perkebunan lebih kuat secara agregat.
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            x_var = st.selectbox("🌱 Komoditas X", numeric_cols, index=0, key="comp_x_var")
        with c2:
            y_var = st.selectbox("🌱 Komoditas Y", numeric_cols, index=1, key="comp_y_var")

        if x_var != y_var:
            fig = px.scatter(
                filtered_df,
                x=x_var,
                y=y_var,
                hover_name="Provinsi",
                title=f"Pola Hubungan Produksi: {x_var} vs {y_var}",
                color_discrete_sequence=["#4ade80"]
            )
            fig = apply_plantation_layout(fig, 550)
            st.plotly_chart(fig, use_container_width=True, key="comp_scatter")

            corr_val = filtered_df[[x_var, y_var]].corr().iloc[0, 1]
            if pd.notna(corr_val):
                strength = "kuat" if abs(corr_val) > 0.6 else "sedang" if abs(corr_val) > 0.3 else "lemah"
                direction = "positif" if corr_val > 0 else "negatif"
                interpretation = ""
                if corr_val > 0.3:
                    interpretation = "Kedua komoditas cenderung tumbuh bersama di wilayah dengan kondisi agroekologi serupa."
                elif corr_val < -0.3:
                    interpretation = "Terdapat pola substitusi wilayah — daerah yang kuat di satu komoditas cenderung lemah di komoditas lainnya."
                else:
                    interpretation = "Kedua komoditas berkembang secara independen, menunjukkan spesialisasi wilayah yang berbeda."

                st.markdown(f"""
                <div class="insight-box">
                <b>🔗 Korelasi Produksi:</b> {corr_val:.3f} ({strength} {direction})<br>
                <i>{interpretation}</i>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Pilih dua komoditas yang berbeda untuk dianalisis hubungannya.")

    with tab3:
        st.markdown("### 🌡️ Matriks Korelasi Antar Komoditas Perkebunan")
        corr = filtered_df[numeric_cols].corr()

        # Palet agro untuk heatmap: cokelat (negatif) → hijau gelap (0) → hijau terang (positif)
        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            zmin=-1,
            zmax=1,
            color_continuous_scale=[
                [0.0, "#7c4f2a"],   # cokelat tanah (korelasi negatif)
                [0.25, "#a0522d"],
                [0.5, "#1a4d2e"],   # hijau gelap (netral)
                [0.75, "#2f855a"],
                [1.0, "#4ade80"]    # hijau terang (korelasi positif)
            ],
            title="Pola Korelasi Produksi Perkebunan"
        )

        fig.update_traces(
            textfont=dict(color="white", size=12),
            xgap=2,
            ygap=2
        )

        fig.update_xaxes(side="bottom", tickangle=30)
        fig.update_yaxes(autorange="reversed")

        fig.update_layout(
            paper_bgcolor="#0d1b12",
            plot_bgcolor="#0d1b12",
            font=dict(color="#e8f0e4"),
            margin=dict(l=40, r=30, t=70, b=40)
        )

        fig.update_coloraxes(
            colorbar=dict(
                title=dict(text="Korelasi", font=dict(color="#e8f0e4")),
                tickfont=dict(color="#e8f0e4"),
                len=0.75
            )
        )

        st.plotly_chart(fig, use_container_width=True, key="corr_heatmap_plantation")

        st.markdown("""
        <div class="insight-box">
        <b>🌱 Interpretasi Agro-Ekologis:</b> Mayoritas korelasi antar komoditas perkebunan bersifat <b>lemah</b>, 
        mencerminkan <b>spesialisasi geografis yang kuat</b>. Setiap wilayah mengembangkan komoditas andalan 
        berdasarkan kesesuaian agroekologi, sejarah budidaya, dan keunggulan komparatif regional.
        Pola ini mengindikasikan bahwa perkebunan Indonesia tidak monokultur massal, melainkan mosaik 
        spesialisasi yang kaya.
        </div>
        """, unsafe_allow_html=True)

    with tab4:
        bench_prov = st.selectbox("🎯 Pilih Wilayah untuk Benchmarking", df["Provinsi"].tolist(), key="bench_prov")
        bench_data = df[df["Provinsi"] == bench_prov].iloc[0]

        bench_df = pd.DataFrame({
            "Komoditas": numeric_cols,
            "Provinsi": [bench_data[c] for c in numeric_cols],
            "Rata_Nasional": [df[c].mean() for c in numeric_cols]
        })
        bench_df["Rasio"] = bench_df["Provinsi"] / bench_df["Rata_Nasional"].replace(0, 0.01)

        color_map_bench = {row["Komoditas"]: get_commodity_color(row["Komoditas"], light=True)
                          for _, row in bench_df.iterrows()}

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=bench_df["Komoditas"],
            y=bench_df["Rasio"],
            marker_color=[color_map_bench.get(c, "#4ade80") for c in bench_df["Komoditas"]],
            name="Rasio Performa"
        ))
        fig.add_hline(y=1, line_dash="dash", line_color="#d4a017")
        fig.update_layout(
            title=f"Performa Relatif {bench_prov} (Garis Emas = Rata-rata Nasional)",
            yaxis_title="Rasio (1.0 = Rata-rata Nasional)"
        )
        fig = apply_plantation_layout(fig, 500)
        st.plotly_chart(fig, use_container_width=True, key="comp_benchmark")

        above_avg = (bench_df["Rasio"] > 1).sum()
        below_avg = (bench_df["Rasio"] < 1).sum()
        strongest = bench_df.loc[bench_df["Rasio"].idxmax()]

        st.markdown(f"""
        <div class="info-box">
        <b>🎯 Ringkasan Benchmark:</b><br>
        • <b>{above_avg}</b> komoditas di atas rata-rata nasional | <b>{below_avg}</b> di bawah<br>
        • Komoditas <b>unggulan relatif</b>: {get_commodity_icon(strongest['Komoditas'])} <b>{strongest['Komoditas']}</b> 
          dengan rasio <b>{strongest['Rasio']:.2f}x</b> rata-rata nasional
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# PAGE 5: SEBARAN WILAYAH PRODUKSI
# =========================================================
elif menu == "🌍 Sebaran Wilayah Produksi":
    st.markdown('<div class="section-title">🌍 Sebaran Wilayah Produksi</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Pemetaan sentra produksi, intensitas wilayah, dan pola distribusi geografis komoditas perkebunan</div>', unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("Tidak ada data untuk filter aktif.")
    else:
        geo_df = filtered_df[["Provinsi", selected_commodity]].copy().sort_values(selected_commodity, ascending=False)
        comm_color = get_commodity_color(selected_commodity, light=True)
        comm_icon = get_commodity_icon(selected_commodity)

        c1, c2 = st.columns([1.2, 1])

        with c1:
            st.markdown(f'<div class="section-title">{comm_icon} Hirarki Sentra Produksi — {selected_commodity}</div>', unsafe_allow_html=True)
            fig = px.bar(
                geo_df.head(min(top_n, len(geo_df))),
                x="Provinsi",
                y=selected_commodity,
                color=selected_commodity,
                text=selected_commodity,
                color_continuous_scale=[COMMODITY_IDENTITY[selected_commodity]["color"], COMMODITY_IDENTITY[selected_commodity]["color_light"]],
                title="Ranking Intensitas Produksi Wilayah"
            )
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_xaxes(tickangle=45)
            fig.update_layout(showlegend=False)
            fig = apply_plantation_layout(fig, 520)
            st.plotly_chart(fig, use_container_width=True, key="geo_ranking")

        with c2:
            st.markdown('<div class="section-title">🎯 Peta Prioritas Wilayah</div>', unsafe_allow_html=True)
            pseudo_geo = geo_df.head(min(15, len(geo_df))).copy()
            pseudo_geo["Rank"] = range(1, len(pseudo_geo) + 1)

            fig2 = px.scatter(
                pseudo_geo,
                x="Rank",
                y=selected_commodity,
                size=selected_commodity,
                hover_name="Provinsi",
                text="Provinsi",
                title="Bubble Prioritas Wilayah Produksi",
                color_discrete_sequence=[comm_color]
            )
            fig2.update_traces(textposition="top center")
            fig2 = apply_plantation_layout(fig2, 520)
            st.plotly_chart(fig2, use_container_width=True, key="geo_bubble")

        # Klasifikasi wilayah
        total_prod = geo_df[selected_commodity].sum()
        if total_prod > 0:
            geo_df["Share"] = (geo_df[selected_commodity] / total_prod) * 100
            geo_df["Kategori"] = pd.cut(
                geo_df["Share"],
                bins=[-1, 1, 5, 15, 100],
                labels=["Wilayah Minor", "Wilayah Penyangga", "Sentra Regional", "Sentra Nasional"]
            )

            kategori_count = geo_df["Kategori"].value_counts()

            st.markdown('<div class="section-title">🏷️ Klasifikasi Wilayah Produksi</div>', unsafe_allow_html=True)
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("🏆 Sentra Nasional", kategori_count.get("Sentra Nasional", 0))
            k2.metric("🌾 Sentra Regional", kategori_count.get("Sentra Regional", 0))
            k3.metric("🌱 Wilayah Penyangga", kategori_count.get("Wilayah Penyangga", 0))
            k4.metric("🍃 Wilayah Minor", kategori_count.get("Wilayah Minor", 0))

        st.markdown(f"""
        <div class="insight-box">
        <b>{comm_icon} Pola Sebaran {selected_commodity}:</b><br>
        Komoditas ini memperlihatkan pola <b>spesialisasi geografis yang jelas</b>, dengan konsentrasi produksi 
        pada sejumlah provinsi tertentu. Wilayah-wilayah dominan berperan sebagai <b>sentra produksi</b> 
        yang menopang pasokan nasional, sementara wilayah lain berperan sebagai <b>penyangga</b> 
        atau memiliki produksi minor. Pola ini mencerminkan kesesuaian agroekologi dan 
        sejarah pengembangan perkebunan di tiap wilayah.
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# PAGE 6: PROYEKSI & MODEL PRODUKSI
# =========================================================
elif menu == "📈 Proyeksi & Model Produksi":
    st.markdown('<div class="section-title">📈 Proyeksi & Model Produksi</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Eksplorasi pola produksi dengan machine learning — forecasting, prediksi, dan segmentasi wilayah perkebunan</div>', unsafe_allow_html=True)

    if filtered_df.shape[0] < 5:
        st.warning("⚠️ Data terlalu sedikit untuk analisis model yang stabil. Silakan perluas filter wilayah.")
    else:
        tab_lr, tab_fc, tab_rf, tab_dt = st.tabs([
            "📐 Model Regresi Linear",
            "📅 Simulasi Panen 2025",
            "🌲 Random Forest",
            "🌳 Decision Tree"
        ])

        with tab_lr:
            st.markdown("### 📐 Model Regresi Hubungan Produksi")
            st.markdown('<div class="section-subtitle">Eksplorasi pola prediktif antar komoditas perkebunan</div>', unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                x_var = st.selectbox("🌱 Komoditas Prediktor (X)", numeric_cols, index=1 if len(numeric_cols) > 1 else 0, key="lr_x")
            with c2:
                y_var = st.selectbox("🌾 Komoditas Target (Y)", numeric_cols, index=0, key="lr_y")

            if x_var == y_var:
                st.warning("Pilih dua komoditas yang berbeda.")
            else:
                model_df = filtered_df[[x_var, y_var]].dropna().copy()
                if len(model_df) < 3:
                    st.warning("Data tidak cukup untuk membangun model regresi.")
                else:
                    X = model_df[[x_var]].values
                    y = model_df[y_var].values

                    lr = LinearRegression()
                    lr.fit(X, y)
                    y_pred = lr.predict(X)

                    mae = mean_absolute_error(y, y_pred)
                    rmse = math.sqrt(mean_squared_error(y, y_pred))
                    r2 = r2_score(y, y_pred)

                    k1, k2, k3 = st.columns(3)
                    k1.metric("MAE", f"{mae:.2f}")
                    k2.metric("RMSE", f"{rmse:.2f}")
                    k3.metric("R²", f"{r2:.4f}")

                    st.markdown(f"""
                    <div class="info-box">
                    <b>📐 Persamaan Model:</b><br>
                    <code>{y_var} = {lr.coef_[0]:.4f} × {x_var} + {lr.intercept_:.4f}</code><br>
                    <i>Model ini mengeksplorasi pola prediktif antar komoditas perkebunan, bukan prediksi absolut.</i>
                    </div>
                    """, unsafe_allow_html=True)

                    plot_df = model_df.copy()
                    plot_df["Prediksi"] = y_pred

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=plot_df[x_var], y=plot_df[y_var], mode="markers", name="Data Wilayah",
                        marker=dict(color="#4ade80", size=10)
                    ))
                    sort_idx = np.argsort(plot_df[x_var].values)
                    fig.add_trace(go.Scatter(
                        x=plot_df[x_var].values[sort_idx], y=plot_df["Prediksi"].values[sort_idx],
                        mode="lines", name="Garis Regresi", line=dict(color="#d4a017", width=3)
                    ))
                    fig.update_layout(title=f"Pola Hubungan: {x_var} → {y_var}", xaxis_title=x_var, yaxis_title=y_var)
                    fig = apply_plantation_layout(fig, 540)
                    st.plotly_chart(fig, use_container_width=True, key="lr_chart")

                    st.markdown("### 🎮 Playground Prediksi")
                    x_input = st.number_input(
                        f"🌱 Masukkan volume {x_var} (Ribu Ton):",
                        min_value=0.0,
                        value=float(np.median(model_df[x_var]))
                    )
                    pred_val = lr.predict(np.array([[x_input]]))[0]
                    st.markdown(f"""
                    <div class="success-box">
                    🌾 Prediksi <b>{y_var}</b> untuk <b>{x_var} = {x_input:.2f}</b> ribu ton: <b>{pred_val:.2f} ribu ton</b>
                    </div>
                    """, unsafe_allow_html=True)

        with tab_fc:
            st.markdown("### 📅 Simulasi Panen 2025")
            st.markdown("""
            <div class="warn-box">
            <b>⚠️ Catatan Metodologi:</b> Dataset perkebunan ini bersifat <i>cross-sectional</i> (1 tahun). 
            Simulasi proyeksi dilakukan berdasarkan <b>skenario growth rate</b>, bukan time-series forecasting historis. 
            Berguna untuk eksplorasi skenario kebijakan, bukan prediksi absolut.
            </div>
            """, unsafe_allow_html=True)

            commodity_target = st.selectbox("🌾 Komoditas untuk Disimulasikan", numeric_cols, key="fc_target")
            growth_rate = st.slider("📈 Growth Rate Panen (%)", min_value=1, max_value=20, value=7, key="fc_growth") / 100

            fc_df = filtered_df[["Provinsi", commodity_target]].copy()
            fc_df["Proyeksi_2025"] = fc_df[commodity_target] * (1 + growth_rate)
            fc_df["Peningkatan"] = fc_df["Proyeksi_2025"] - fc_df[commodity_target]

            top_fc = fc_df.sort_values(commodity_target, ascending=False).head(min(top_n, len(fc_df))).copy()
            comm_color = get_commodity_color(commodity_target, light=True)

            fig = go.Figure()
            fig.add_trace(go.Bar(x=top_fc["Provinsi"], y=top_fc[commodity_target], name="Panen 2024", marker_color="#c7815e"))
            fig.add_trace(go.Bar(x=top_fc["Provinsi"], y=top_fc["Proyeksi_2025"], name="Proyeksi 2025", marker_color="#4ade80"))
            fig.update_layout(barmode="group", title=f"Perbandingan Panen {commodity_target}: 2024 vs 2025")
            fig = apply_plantation_layout(fig, 550)
            st.plotly_chart(fig, use_container_width=True, key="fc_chart")

            total_now = fc_df[commodity_target].sum()
            total_fc = fc_df["Proyeksi_2025"].sum()

            k1, k2, k3 = st.columns(3)
            k1.metric("🌾 Panen 2024", format_num(total_now), "Ribu Ton")
            k2.metric("📅 Proyeksi 2025", format_num(total_fc), "Ribu Ton")
            k3.metric("📈 Potensi Peningkatan", format_num(total_fc - total_now), f"+{growth_rate*100:.0f}%")

            st.dataframe(fc_df.sort_values("Proyeksi_2025", ascending=False), use_container_width=True)

        with tab_rf:
            st.markdown("### 🌲 Random Forest — Prediksi Volume Produksi")
            st.markdown('<div class="section-subtitle">Model ensemble berbasis hutan keputusan untuk eksplorasi pola produksi</div>', unsafe_allow_html=True)

            target_rf = st.selectbox("🎯 Target Prediksi", numeric_cols, key="rf_target")
            feature_cols = [c for c in numeric_cols if c != target_rf]
            rf_df = filtered_df[feature_cols + [target_rf]].dropna().copy()

            if len(rf_df) < 8:
                st.warning("Data terlalu sedikit untuk Random Forest yang stabil.")
            else:
                X = rf_df[feature_cols]
                y = rf_df[target_rf]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                rf = RandomForestRegressor(n_estimators=150, max_depth=5, random_state=42)
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)

                mae = mean_absolute_error(y_test, y_pred)
                rmse = math.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)

                k1, k2, k3 = st.columns(3)
                k1.metric("MAE", f"{mae:.2f}")
                k2.metric("RMSE", f"{rmse:.2f}")
                k3.metric("R²", f"{r2:.4f}")

                importance = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False).reset_index()
                importance.columns = ["Komoditas", "Kepentingan"]

                color_map_rf = {row["Komoditas"]: get_commodity_color(row["Komoditas"], light=True)
                               for _, row in importance.iterrows()}

                fig = px.bar(
                    importance.sort_values("Kepentingan", ascending=True),
                    x="Kepentingan",
                    y="Komoditas",
                    orientation="h",
                    text="Kepentingan",
                    color="Komoditas",
                    color_discrete_map=color_map_rf,
                    title=f"Komoditas Paling Berpengaruh terhadap {target_rf}"
                )
                fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                fig.update_layout(showlegend=False)
                fig = apply_plantation_layout(fig, 500)
                st.plotly_chart(fig, use_container_width=True, key="rf_chart")

                top_feat = importance.iloc[-1]
                st.markdown(f"""
                <div class="insight-box">
                🌱 <b>{top_feat['Komoditas']}</b> menjadi komoditas paling berpengaruh dalam memprediksi produksi 
                <b>{target_rf}</b> dengan skor kepentingan <b>{top_feat['Kepentingan']:.3f}</b>. 
                Ini menunjukkan adanya pola agro-ekologis atau geografis yang berkaitan antara kedua komoditas.
                </div>
                """, unsafe_allow_html=True)

        with tab_dt:
            st.markdown("### 🌳 Decision Tree — Segmentasi Wilayah Produksi")
            st.markdown('<div class="section-subtitle">Model pohon keputusan untuk memahami pola segmentasi produksi komoditas</div>', unsafe_allow_html=True)

            target_dt = st.selectbox("🎯 Target Prediksi", numeric_cols, key="dt_target")
            feature_cols_dt = [c for c in numeric_cols if c != target_dt]
            dt_df = filtered_df[feature_cols_dt + [target_dt]].dropna().copy()

            if len(dt_df) < 8:
                st.warning("Data terlalu sedikit untuk Decision Tree yang stabil.")
            else:
                X = dt_df[feature_cols_dt]
                y = dt_df[target_dt]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                dt = DecisionTreeRegressor(max_depth=3, random_state=42)
                dt.fit(X_train, y_train)
                y_pred = dt.predict(X_test)

                mae = mean_absolute_error(y_test, y_pred)
                rmse = math.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)

                k1, k2, k3 = st.columns(3)
                k1.metric("MAE", f"{mae:.2f}")
                k2.metric("RMSE", f"{rmse:.2f}")
                k3.metric("R²", f"{r2:.4f}")

                importance = pd.Series(dt.feature_importances_, index=feature_cols_dt).sort_values(ascending=False).reset_index()
                importance.columns = ["Komoditas", "Kepentingan"]

                color_map_dt = {row["Komoditas"]: get_commodity_color(row["Komoditas"], light=True)
                               for _, row in importance.iterrows()}

                fig = px.bar(
                    importance.sort_values("Kepentingan", ascending=True),
                    x="Kepentingan",
                    y="Komoditas",
                    orientation="h",
                    text="Kepentingan",
                    color="Komoditas",
                    color_discrete_map=color_map_dt,
                    title=f"Pembagi Utama Segmentasi {target_dt}"
                )
                fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                fig.update_layout(showlegend=False)
                fig = apply_plantation_layout(fig, 500)
                st.plotly_chart(fig, use_container_width=True, key="dt_chart")

                st.markdown(f"""
                <div class="insight-box">
                🌳 Model Decision Tree dengan kedalaman 3 level menunjukkan bahwa <b>{importance.iloc[-1]['Komoditas']}</b> 
                menjadi variabel pembagi utama segmentasi wilayah produksi <b>{target_dt}</b>. 
                Model ini membantu memahami bagaimana pola segmentasi alami wilayah perkebunan Indonesia.
                </div>
                """, unsafe_allow_html=True)

# =========================================================
# PAGE 7: INSIGHT & STRATEGI PERKEBUNAN
# =========================================================
elif menu == "🧠 Insight & Strategi Perkebunan":
    st.markdown('<div class="section-title">🧠 Insight & Strategi Perkebunan</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Temuan strategis dan rekomendasi implementatif untuk pengembangan sektor perkebunan Indonesia</div>', unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("Tidak ada data untuk filter aktif.")
    else:
        total_by_comm = filtered_df[numeric_cols].sum().sort_values(ascending=False)
        dominant_comm = total_by_comm.index[0]
        dominant_val = total_by_comm.iloc[0]
        dom_icon = get_commodity_icon(dominant_comm)

        top_prov, top_val = top_province_for_commodity(filtered_df, selected_commodity)
        comm_icon = get_commodity_icon(selected_commodity)
        total_selected = filtered_df[selected_commodity].sum()
        share_top = (top_val / total_selected * 100) if total_selected > 0 else 0

        diverse_prov, diversity_score = province_with_most_diverse(filtered_df)
        top5_share = add_total_production(filtered_df).sort_values("Total Produksi", ascending=False).head(5)["Total Produksi"].sum() / max(1, add_total_production(filtered_df)["Total Produksi"].sum()) * 100

        st.markdown("### 🌾 Insight Strategis Nasional")

        insights = [
            f"{dom_icon} <b>{dominant_comm}</b> merupakan komoditas perkebunan paling dominan secara nasional dengan volume <b>{format_ton(dominant_val)} ribu ton</b>, menjadikannya tulang punggung sektor perkebunan Indonesia.",

            f"{comm_icon} Pada komoditas <b>{selected_commodity}</b>, <b>{top_prov}</b> berperan sebagai sentra utama dengan kontribusi <b>{share_top:.1f}%</b> terhadap total produksi.",

            f"🗺️ <b>Konsentrasi Wilayah:</b> Top 5 provinsi menguasai <b>{top5_share:.1f}%</b> total produksi perkebunan nasional, menunjukkan pola konsentrasi spasial yang perlu diwaspadai dari sisi risiko iklim dan pasar.",

            f"🌱 <b>{diverse_prov}</b> teridentifikasi sebagai provinsi dengan portofolio komoditas paling terdiversifikasi (<b>{int(diversity_score)}</b> komoditas aktif), menjadi contoh ketahanan struktural yang baik.",

            "🌳 Korelasi lemah antar komoditas menunjukkan pola <b>spesialisasi geografis yang sehat</b>, di mana setiap wilayah mengembangkan keunggulan komparatifnya masing-masing — fondasi penting untuk ketahanan sektor perkebunan nasional."
        ]

        for i, ins in enumerate(insights, start=1):
            st.markdown(f'<div class="insight-box"><b>#{i}.</b> {ins}</div>', unsafe_allow_html=True)

        st.markdown("### 🚀 Strategi Implementatif")
        recs = generate_recommendations(filtered_df, selected_commodity)
        for i, rec in enumerate(recs, start=1):
            st.markdown(f'<div class="success-box"><b>#{i}.</b> {rec}</div>', unsafe_allow_html=True)

# =========================================================
# PAGE 8: DATA & EKSPOR
# =========================================================
elif menu == "📦 Data & Ekspor":
    st.markdown('<div class="section-title">📦 Data & Ekspor Perkebunan</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Dokumentasi dataset, kualitas data, dan ekspor untuk kebutuhan analisis lanjutan</div>', unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("Tidak ada data untuk diekspor.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📋 Total Observasi", filtered_df.shape[0])
        c2.metric("🌾 Variabel Komoditas", len(numeric_cols))
        c3.metric("✅ Missing Values", int(filtered_df.isnull().sum().sum()))
        c4.metric("🔄 Data Duplikat", int(filtered_df.duplicated().sum()))

        tab1, tab2, tab3 = st.tabs(["📋 Dataset Mentah", "📊 Statistik Deskriptif", "🧹 Kualitas Data"])

        with tab1:
            st.markdown('<div class="section-subtitle">Data produksi perkebunan per provinsi (Ribu Ton)</div>', unsafe_allow_html=True)
            st.dataframe(filtered_df, use_container_width=True)

        with tab2:
            st.markdown('<div class="section-subtitle">Statistik deskriptif komoditas perkebunan</div>', unsafe_allow_html=True)
            desc = filtered_df[numeric_cols].describe().T
            desc["range"] = desc["max"] - desc["min"]
            st.dataframe(desc, use_container_width=True)

        with tab3:
            st.markdown("### 🧹 Laporan Kualitas Data")
            duplicate_count = filtered_df.duplicated().sum()
            zero_rows = (filtered_df[numeric_cols].sum(axis=1) == 0).sum()

            st.markdown(f"- ✅ **Missing values:** {int(filtered_df.isnull().sum().sum())}")
            st.markdown(f"- ✅ **Baris duplikat:** {int(duplicate_count)}")
            st.markdown(f"- ⚠️ **Wilayah tanpa produksi:** {int(zero_rows)} (termasuk wilayah urban seperti DKI Jakarta)")

            st.markdown("""
            <div class="warn-box">
            <b>📌 Catatan Kualitas Data:</b><br>
            Outlier yang terdeteksi (seperti Riau untuk sawit, Jatim untuk tebu) <b>bukan kesalahan data</b>, 
            melainkan representasi dari <b>sentra produksi riil</b> sektor perkebunan Indonesia. 
            Data ini bersumber dari laporan resmi BPS dan dapat dipercaya untuk analisis strategis.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📥 Ekspor Dataset")
        export_df = add_total_production(filtered_df)
        st.download_button(
            label="⬇️ Ekspor Dataset Terfilter (CSV)",
            data=export_csv(export_df),
            file_name="data_perkebunan_terfilter.csv",
            mime="text/csv"
        )

        st.markdown("### 📅 Ekspor Simulasi Panen 2025")
        exp_comm = st.selectbox("🌾 Komoditas untuk Diekspor", numeric_cols, key="exp_comm")
        exp_growth = st.slider("📈 Growth rate simulasi (%)", 1, 20, 7, key="exp_growth") / 100

        export_fc = filtered_df[["Provinsi", exp_comm]].copy()
        export_fc["Proyeksi_2025"] = export_fc[exp_comm] * (1 + exp_growth)
        export_fc["Peningkatan"] = export_fc["Proyeksi_2025"] - export_fc[exp_comm]

        st.dataframe(export_fc, use_container_width=True)

        st.download_button(
            label=f"⬇️ Ekspor Proyeksi {exp_comm} 2025 (CSV)",
            data=export_csv(export_fc),
            file_name=f"proyeksi_{exp_comm.lower().replace(' ', '_')}_2025.csv",
            mime="text/csv"
        )

        st.markdown("### 📊 Ekspor Statistik Deskriptif")
        st.download_button(
            label="⬇️ Ekspor Statistik Deskriptif (CSV)",
            data=export_csv(desc),
            file_name="statistik_perkebunan.csv",
            mime="text/csv"
        )

# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="footer-box">
    <h3 style="color: #4ade80;">🌴 Dashboard Intelijen Komoditas Perkebunan Indonesia</h3>
    <p style="color: #c7d2c0;">UAS Pengenalan Sains Data — Visualisasi Data & Analisis Data Dasar</p>
    <p style="color: #93a39a; font-size: 0.85rem; margin-top: 1rem;">
        Dibangun dengan Streamlit, Plotly, dan Scikit-Learn untuk mendukung perencanaan strategis sektor perkebunan Indonesia
    </p>
    <p style="color: #93a39a; font-size: 0.8rem;">
        © 2026 | Sumber Data: BPS — Produksi Tanaman Perkebunan Menurut Provinsi, 2024
    </p>
</div>
""", unsafe_allow_html=True)
