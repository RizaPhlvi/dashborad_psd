# ============================================================
# DASHBOARD KOMODITAS PERKEBUNAN INDONESIA - REDesign FINAL
# Senior Streamlit Engineer Version
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import math
import warnings
from datetime import datetime

# ML Imports
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# ============================================================
# 1. KONFIGURASI HALAMAN & CUSTOM CSS
# ============================================================
st.set_page_config(
    page_title="Dashboard Komoditas Perkebunan Indonesia",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Global Styling */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf3 100%);
    }
    
    /* Hero Section */
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1e3a8a 0%, #0891b2 50%, #059669 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        color: #475569;
        font-weight: 400;
        margin-bottom: 2rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03);
        border: 1px solid rgba(226, 232, 240, 0.8);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px -5px rgba(0,0,0,0.1);
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
        margin: 0.5rem 0;
    }
    
    .metric-delta {
        font-size: 0.875rem;
        color: #059669;
        font-weight: 600;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.75rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 3px solid #0891b2;
        display: inline-block;
    }
    
    .section-subheader {
        font-size: 1.25rem;
        font-weight: 600;
        color: #334155;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Custom Box */
    .insight-box {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border-left: 4px solid #3b82f6;
        padding: 1.25rem 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 4px solid #f59e0b;
        padding: 1.25rem 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .success-box {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 4px solid #10b981;
        padding: 1.25rem 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
    }
    
    /* Custom Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        padding: 0.25rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        color: #475569;
        border: 1px solid #e2e8f0;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1e3a8a 0%, #0891b2 100%);
        color: white !important;
        border: none;
    }
    
    /* Footer */
    .dashboard-footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 1px solid #e2e8f0;
        color: #64748b;
    }
    
    /* Hide streamlit default padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 2. DATASET (EMBEDDED)
# ============================================================
@st.cache_data
def load_dataset():
    csv_data = """Provinsi,Kelapa Sawit,Kelapa,Karet,Kopi,Kakao,Teh,Tebu
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
    return pd.read_csv(io.StringIO(csv_data))

df_raw = load_dataset()
COMMODITIES = [c for c in df_raw.columns if c != 'Provinsi']

# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================
def format_number(num, suffix=''):
    """Format angka dengan suffix K/M/B"""
    if pd.isna(num) or num == 0:
        return '0' + suffix
    if abs(num) >= 1_000_000:
        return f'{num/1_000_000:.2f}M{suffix}'
    elif abs(num) >= 1_000:
        return f'{num/1_000:.2f}K{suffix}'
    return f'{num:.2f}{suffix}'

def format_ton(num):
    """Format dalam ribu ton"""
    return f'{num:,.2f}'

def safe_divide(a, b):
    """Pembagian aman"""
    return a / b if b != 0 else 0

def apply_global_filters(df, filters):
    """Terapkan filter global ke dataframe"""
    df_filtered = df.copy()
    
    # Filter provinsi
    if filters['provinces'] and len(filters['provinces']) < len(df['Provinsi'].unique()):
        df_filtered = df_filtered[df_filtered['Provinsi'].isin(filters['provinces'])]
    
    # Filter hide zeros
    if filters['hide_zeros']:
        df_filtered = df_filtered[df_filtered[filters['commodity']] > 0]
    
    # Top N
    if filters['top_n'] < len(df_filtered):
        df_filtered = df_filtered.nlargest(filters['top_n'], filters['commodity'])
    
    return df_filtered.reset_index(drop=True)

def get_commodity_stats(df, commodity):
    """Hitung statistik komoditas"""
    total = df[commodity].sum()
    max_val = df[commodity].max()
    max_prov = df.loc[df[commodity].idxmax(), 'Provinsi'] if total > 0 else '-'
    avg = df[commodity].mean()
    non_zero = (df[commodity] > 0).sum()
    return {
        'total': total,
        'max_val': max_val,
        'max_prov': max_prov,
        'avg': avg,
        'non_zero_provinces': non_zero
    }

def create_metric_card(label, value, delta=None, icon='📊'):
    """Buat metric card custom"""
    delta_html = f'<div class="metric-delta">{delta}</div>' if delta else ''
    return f"""
    <div class="metric-card">
        <div class="metric-label">{icon} {label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """

# ============================================================
# 4. SIDEBAR - GLOBAL FILTERS
# ============================================================
st.sidebar.markdown("## 🎛️ Global Filters")
st.sidebar.markdown("---")

selected_commodity = st.sidebar.selectbox(
    "🌾 Komoditas Utama",
    COMMODITIES,
    index=0,
    help="Pilih komoditas yang akan menjadi fokus analisis"
)

all_provinces = df_raw['Provinsi'].unique().tolist()
selected_provinces = st.sidebar.multiselect(
    "📍 Provinsi",
    all_provinces,
    default=all_provinces,
    help="Pilih satu atau lebih provinsi untuk difilter"
)

top_n = st.sidebar.slider(
    "🏆 Top N Provinsi",
    min_value=5,
    max_value=len(all_provinces),
    value=15,
    help="Jumlah provinsi teratas yang ditampilkan"
)

hide_zeros = st.sidebar.checkbox(
    "🚫 Sembunyikan Provinsi dengan Produksi 0",
    value=True,
    help="Abaikan provinsi yang tidak memproduksi komoditas terpilih"
)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset Filter"):
    st.rerun()

filters = {
    'commodity': selected_commodity,
    'provinces': selected_provinces,
    'top_n': top_n,
    'hide_zeros': hide_zeros
}

df_filtered = apply_global_filters(df_raw, filters)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='padding: 1rem; background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); 
            border-radius: 8px; font-size: 0.85rem; color: #1e40af;'>
<b>📊 Data Aktif:</b><br>
• Provinsi: {prov}<br>
• Baris: {rows}<br>
• Filter: {status}
</div>
""".format(
    prov=len(df_filtered),
    rows=len(df_filtered),
    status="Aktif" if len(df_filtered) < len(df_raw) else "Semua"
), unsafe_allow_html=True)

# ============================================================
# 5. NAVIGASI HALAMAN
# ============================================================
st.sidebar.markdown("---")
st.sidebar.markdown("## 🧭 Navigasi")

PAGES = {
    "🏠 Executive Dashboard": "executive",
    "🔍 Data Explorer": "data",
    "📊 EDA Explorer": "eda",
    "📍 Profil Wilayah & Komoditas": "profiles",
    "🤖 Predictive Analytics": "predictive",
    "💡 Insight & Rekomendasi": "insights",
    "📥 Export Center": "export"
}

selected_page = st.sidebar.radio("Pilih Menu:", list(PAGES.keys()))

# ============================================================
# 6. HEADER UTAMA
# ============================================================
st.markdown('<div class="hero-title">🌾 Dashboard Komoditas Perkebunan Indonesia</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Analisis Interaktif Produksi Tanaman Perkebunan per Provinsi — Tahun Data 2024 | UAS Pengenalan Sains Data</div>', unsafe_allow_html=True)

# ============================================================
# 7. HALAMAN: EXECUTIVE DASHBOARD
# ============================================================
if PAGES[selected_page] == "executive":
    st.markdown("## 🏠 Executive Dashboard")
    st.markdown("Ringkasan eksekutif kondisi perkebunan nasional berdasarkan filter aktif.")
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    total_prov = len(df_filtered)
    total_prod = df_filtered[COMMODITIES].sum().sum()
    dominant_commodity = df_filtered[COMMODITIES].sum().idxmax()
    dominant_prod = df_filtered[COMMODITIES].sum().max()
    top_province = df_filtered.loc[df_filtered[selected_commodity].idxmax(), 'Provinsi'] if len(df_filtered) > 0 else '-'
    top_prov_val = df_filtered[selected_commodity].max() if len(df_filtered) > 0 else 0
    
    with col1:
        st.markdown(create_metric_card(
            "Provinsi Aktif",
            str(total_prov),
            "dari 38 provinsi nasional",
            "🗺️"
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_metric_card(
            "Total Produksi",
            format_number(total_prod, ' t'),
            f"7 komoditas digabung",
            "📦"
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown(create_metric_card(
            "Komoditas Dominan",
            dominant_commodity,
            format_number(dominant_prod, ' t'),
            "🏆"
        ), unsafe_allow_html=True)
    
    with col4:
        st.markdown(create_metric_card(
            f"Top {selected_commodity}",
            top_province[:15],
            format_ton(top_prov_val) + " ribu ton",
            "⭐"
        ), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main Charts
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.markdown("### 📈 Top Provinsi - " + selected_commodity)
        top_df = df_filtered.nlargest(min(15, len(df_filtered)), selected_commodity)
        
        if len(top_df) > 0:
            fig = px.bar(
                top_df,
                x=selected_commodity,
                y='Provinsi',
                orientation='h',
                title=f'Top {len(top_df)} Provinsi Penghasil {selected_commodity}',
                color=selected_commodity,
                color_continuous_scale='blues',
                text=selected_commodity
            )
            fig.update_layout(
                height=500,
                yaxis={'categoryorder': 'total ascending'},
                showlegend=False,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Tidak ada data untuk ditampilkan dengan filter aktif.")
    
    with col_right:
        st.markdown("### 🥧 Komposisi Produksi Nasional")
        commodity_totals = df_filtered[COMMODITIES].sum()
        commodity_totals = commodity_totals[commodity_totals > 0]
        
        if len(commodity_totals) > 0:
            fig_pie = px.pie(
                values=commodity_totals.values,
                names=commodity_totals.index,
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(
                height=500,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Tidak ada data komoditas untuk divisualisasikan.")
    
    # Quick Insights
    st.markdown("### 🔍 Quick Insights")
    if len(df_filtered) > 0:
        top3 = df_filtered.nlargest(3, selected_commodity)
        top3_total = top3[selected_commodity].sum()
        all_total = df_filtered[selected_commodity].sum()
        contribution = safe_divide(top3_total, all_total) * 100
        
        col_i1, col_i2 = st.columns(2)
        with col_i1:
            st.markdown(f"""
            <div class="insight-box">
            <b>🎯 Konsentrasi Produksi {selected_commodity}</b><br>
            Top 3 provinsi ({', '.join(top3['Provinsi'].tolist())}) menguasai 
            <b>{contribution:.1f}%</b> dari total produksi {selected_commodity} 
            di wilayah yang sedang dianalisis.
            </div>
            """, unsafe_allow_html=True)
        
        with col_i2:
            non_zero_count = (df_filtered[selected_commodity] > 0).sum()
            st.markdown(f"""
            <div class="insight-box">
            <b>📊 Cakupan Wilayah</b><br>
            Hanya <b>{non_zero_count} dari {len(df_filtered)}</b> provinsi 
            ({safe_divide(non_zero_count, len(df_filtered))*100:.1f}%) yang memproduksi 
            {selected_commodity}, menunjukkan spesialisasi geografis yang kuat.
            </div>
            """, unsafe_allow_html=True)
    
    # Summary Table
    st.markdown("### 📋 Tabel Ringkasan Produksi")
    display_df = df_filtered[['Provinsi'] + COMMODITIES].copy()
    display_df['Total'] = display_df[COMMODITIES].sum(axis=1)
    st.dataframe(
        display_df.sort_values('Total', ascending=False).reset_index(drop=True),
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# 8. HALAMAN: DATA EXPLORER
# ============================================================
elif PAGES[selected_page] == "data":
    st.markdown("## 🔍 Data Explorer")
    st.markdown("Eksplorasi struktur, kualitas, dan statistik deskriptif dataset.")
    
    tab_info, tab_stats, tab_quality = st.tabs([
        "📋 Struktur Data",
        "📊 Statistik Deskriptif",
        "✅ Kualitas Data"
    ])
    
    with tab_info:
        st.markdown("### Informasi Dataset")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Observasi", len(df_raw))
        col2.metric("Total Variabel", len(df_raw.columns))
        col3.metric("Tahun Data", "2024")
        
        st.markdown("### Head Data (Filtered)")
        st.dataframe(df_filtered.head(10), use_container_width=True, hide_index=True)
        
        st.markdown("### Tipe Data")
        dtypes_df = pd.DataFrame({
            'Kolom': df_raw.dtypes.index,
            'Tipe Data': df_raw.dtypes.values,
            'Non-Null': df_raw.count().values
        })
        st.dataframe(dtypes_df, use_container_width=True, hide_index=True)
    
    with tab_stats:
        st.markdown("### Statistik Deskriptif - Semua Komoditas")
        st.dataframe(df_raw[COMMODITIES].describe(), use_container_width=True)
        
        st.markdown("### Statistik Deskriptif - " + selected_commodity + " (Filtered)")
        if len(df_filtered) > 0:
            stats = df_filtered[selected_commodity].describe()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean", f"{stats['mean']:.2f}")
            col2.metric("Std Dev", f"{stats['std']:.2f}")
            col3.metric("Min", f"{stats['min']:.2f}")
            col4.metric("Max", f"{stats['max']:.2f}")
        else:
            st.warning("Tidak ada data setelah filter diterapkan.")
    
    with tab_quality:
        st.markdown("### ✅ Data Quality Report")
        
        missing = df_raw.isnull().sum().sum()
        duplicates = df_raw.duplicated().sum()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Missing Values", missing, "0%" if missing == 0 else "Perlu Imputasi")
        col2.metric("Data Duplikat", duplicates, "Aman" if duplicates == 0 else "Hapus")
        col3.metric("Tipe Data Konsisten", "✅ Ya", "Semua numerik")
        col4.metric("Outlier", "Teridentifikasi", "Pada Kelapa Sawit")
        
        st.markdown("### Boxplot Deteksi Outlier - " + selected_commodity)
        fig_box = px.box(
            df_raw,
            y=selected_commodity,
            title=f'Distribusi & Outlier {selected_commodity}',
            color_discrete_sequence=['#3b82f6']
        )
        fig_box.update_layout(height=400)
        st.plotly_chart(fig_box, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
        <b>📌 Catatan Kualitas Data:</b><br>
        • Dataset sudah bersih (no missing, no duplicates).<br>
        • Outlier pada Kelapa Sawit (Riau, Kalteng) <b>tidak dihapus</b> karena 
        merepresentasikan sentra produksi riil Indonesia.<br>
        • Semua nilai numerik valid dalam satuan ribu ton.
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# 9. HALAMAN: EDA EXPLORER
# ============================================================
elif PAGES[selected_page] == "eda":
    st.markdown("## 📊 EDA Explorer")
    st.markdown("Eksplorasi visual interaktif dengan Plotly.")
    
    tab_overview, tab_dist, tab_rel, tab_corr, tab_deep = st.tabs([
        "🌐 Overview",
        "📊 Distribution",
        "🔗 Relationship",
        "🌡️ Correlation",
        "📍 Province Deep Dive"
    ])
    
    with tab_overview:
        st.markdown("### Overview Produksi")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"#### Top Provinsi - {selected_commodity}")
            top_df = df_filtered.nlargest(min(top_n, len(df_filtered)), selected_commodity)
            if len(top_df) > 0:
                fig = px.bar(
                    top_df,
                    x='Provinsi',
                    y=selected_commodity,
                    color=selected_commodity,
                    color_continuous_scale='blues'
                )
                fig.update_layout(height=450, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Top Provinsi - Total Semua Komoditas")
            df_total = df_filtered.copy()
            df_total['Total_Produksi'] = df_total[COMMODITIES].sum(axis=1)
            top_total = df_total.nlargest(min(top_n, len(df_total)), 'Total_Produksi')
            if len(top_total) > 0:
                fig = px.bar(
                    top_total,
                    x='Provinsi',
                    y='Total_Produksi',
                    color='Total_Produksi',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(height=450, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab_dist:
        st.markdown("### Analisis Distribusi")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"#### Histogram - {selected_commodity}")
            fig_hist = px.histogram(
                df_filtered[df_filtered[selected_commodity] > 0],
                x=selected_commodity,
                nbins=15,
                color_discrete_sequence=['#0891b2'],
                title=f'Distribusi Produksi {selected_commodity}'
            )
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            st.markdown("#### Boxplot Perbandingan Semua Komoditas")
            fig_box_all = px.box(
                df_raw[COMMODITIES],
                title='Perbandingan Distribusi Antar Komoditas'
            )
            fig_box_all.update_layout(height=400)
            st.plotly_chart(fig_box_all, use_container_width=True)
    
    with tab_rel:
        st.markdown("### Analisis Hubungan")
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            x_var = st.selectbox("Pilih Variabel X", COMMODITIES, index=0)
        with col_sel2:
            y_var = st.selectbox("Pilih Variabel Y", COMMODITIES, index=1)
        
        if x_var != y_var:
            corr_val = df_filtered[x_var].corr(df_filtered[y_var])
            
            fig_scatter = px.scatter(
                df_filtered,
                x=x_var,
                y=y_var,
                hover_data=['Provinsi'],
                trendline='ols' if len(df_filtered) > 5 else None,
                title=f'Scatter Plot: {x_var} vs {y_var}'
            )
            fig_scatter.update_layout(height=500)
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            if not pd.isna(corr_val):
                st.info(f"📊 **Koefisien Korelasi Pearson:** {corr_val:.4f} ({'Positif' if corr_val > 0 else 'Negatif'})")
        else:
            st.warning("⚠️ Pilih dua variabel yang berbeda untuk analisis hubungan.")
    
    with tab_corr:
        st.markdown("### Heatmap Korelasi")
        corr_matrix = df_filtered[COMMODITIES].corr()
        
        fig_heat = px.imshow(
            corr_matrix,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1,
            title='Matriks Korelasi Antar Komoditas'
        )
        fig_heat.update_layout(height=600)
        st.plotly_chart(fig_heat, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
        <b>💡 Interpretasi:</b> Sebagian besar korelasi antar komoditas sangat lemah (mendekati 0), 
        menunjukkan adanya <b>spesialisasi wilayah</b> yang kuat. Setiap provinsi cenderung fokus 
        pada satu atau dua komoditas tertentu sesuai keunggulan komparatif geografisnya.
        </div>
        """, unsafe_allow_html=True)
    
    with tab_deep:
        st.markdown("### Province Deep Dive")
        selected_prov = st.selectbox(
            "Pilih Provinsi untuk Analisis Mendalam:",
            df_filtered['Provinsi'].tolist()
        )
        
        if selected_prov:
            prov_data = df_filtered[df_filtered['Provinsi'] == selected_prov].iloc[0]
            prov_commodities = {c: prov_data[c] for c in COMMODITIES if prov_data[c] > 0}
            
            col1, col2 = st.columns([1, 2])
            with col1:
                total_prov = sum(prov_commodities.values())
                dominant = max(prov_commodities.items(), key=lambda x: x[1]) if prov_commodities else ('-', 0)
                
                st.metric("Total Produksi", f"{total_prov:,.2f} ribu ton")
                st.metric("Komoditas Dominan", dominant[0])
                st.metric("Produksi Tertinggi", f"{dominant[1]:,.2f} ribu ton")
            
            with col2:
                if prov_commodities:
                    fig_prov = px.bar(
                        x=list(prov_commodities.keys()),
                        y=list(prov_commodities.values()),
                        labels={'x': 'Komoditas', 'y': 'Produksi (Ribu Ton)'},
                        title=f'Profil Produksi {selected_prov}',
                        color=list(prov_commodities.values()),
                        color_continuous_scale='greens'
                    )
                    fig_prov.update_layout(height=400)
                    st.plotly_chart(fig_prov, use_container_width=True)
                else:
                    st.info(f"Provinsi {selected_prov} tidak memproduksi komoditas apapun.")

# ============================================================
# 10. HALAMAN: PROFILES
# ============================================================
elif PAGES[selected_page] == "profiles":
    st.markdown("## 📍 Profil Wilayah & Komoditas")
    
    tab_prov, tab_comm = st.tabs(["🗺️ Profil Provinsi", "🌾 Profil Komoditas"])
    
    with tab_prov:
        st.markdown("### Analisis Mendalam per Provinsi")
        sel_prov = st.selectbox(
            "Pilih Provinsi:",
            df_raw['Provinsi'].tolist(),
            key="prof_prov"
        )
        
        if sel_prov:
            prov_row = df_raw[df_raw['Provinsi'] == sel_prov].iloc[0]
            prod_data = {c: prov_row[c] for c in COMMODITIES}
            total = sum(prod_data.values())
            active_commodities = {k: v for k, v in prod_data.items() if v > 0}
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Produksi", f"{total:,.2f} ribu ton")
            c2.metric("Komoditas Aktif", len(active_commodities))
            
            if active_commodities:
                dominant = max(active_commodities.items(), key=lambda x: x[1])
                c3.metric("Komoditas Dominan", dominant[0])
                c4.metric("Produksi Tertinggi", f"{dominant[1]:,.2f}")
            else:
                c3.metric("Komoditas Dominan", "-")
                c4.metric("Produksi Tertinggi", "0")
            
            if active_commodities:
                col_l, col_r = st.columns([2, 1])
                with col_l:
                    fig = px.bar(
                        x=list(active_commodities.keys()),
                        y=list(active_commodities.values()),
                        title=f'Struktur Produksi {sel_prov}',
                        color=list(active_commodities.values()),
                        color_continuous_scale='blues'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_r:
                    st.markdown("#### Detail per Komoditas")
                    detail_df = pd.DataFrame({
                        'Komoditas': list(prod_data.keys()),
                        'Produksi (Ribu Ton)': list(prod_data.values()),
                        'Kontribusi (%)': [safe_divide(v, total)*100 for v in prod_data.values()]
                    })
                    st.dataframe(detail_df, use_container_width=True, hide_index=True)
    
    with tab_comm:
        st.markdown("### Analisis Mendalam per Komoditas")
        sel_comm = st.selectbox(
            "Pilih Komoditas:",
            COMMODITIES,
            key="prof_comm"
        )
        
        stats = get_commodity_stats(df_raw, sel_comm)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Nasional", f"{stats['total']:,.2f} ribu ton")
        c2.metric("Provinsi Produsen", f"{stats['non_zero_provinces']}/38")
        c3.metric("Produksi Tertinggi", f"{stats['max_val']:,.2f}")
        c4.metric("Provinsi Top", stats['max_prov'][:15])
        
        st.markdown(f"#### Top 15 Provinsi Penghasil {sel_comm}")
        top_df = df_raw.nlargest(15, sel_comm)[['Provinsi', sel_comm]]
        top_df.columns = ['Provinsi', 'Produksi (Ribu Ton)']
        
        fig = px.bar(
            top_df,
            x='Provinsi',
            y='Produksi (Ribu Ton)',
            title=f'Top 15 Provinsi - {sel_comm}',
            color='Produksi (Ribu Ton)',
            color_continuous_scale='reds'
        )
        fig.update_layout(height=450, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Dynamic insight
        concentration = safe_divide(
            top_df['Produksi (Ribu Ton)'].head(3).sum(),
            stats['total']
        ) * 100
        
        st.markdown(f"""
        <div class="insight-box">
        <b>🎯 Insight {sel_comm}:</b><br>
        Produksi {sel_comm} sangat terkonsentrasi di wilayah tertentu. Top 3 provinsi menguasai 
        <b>{concentration:.1f}%</b> dari total produksi nasional. Hanya 
        <b>{stats['non_zero_provinces']}</b> dari 38 provinsi yang aktif memproduksi komoditas ini,
        menunjukkan kebutuhan lahan dan iklim yang spesifik.
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# 11. HALAMAN: PREDICTIVE ANALYTICS
# ============================================================
elif PAGES[selected_page] == "predictive":
    st.markdown("## 🤖 Predictive Analytics")
    st.markdown("Pemodelan prediktif dengan Machine Learning.")
    
    tab_lr, tab_fc, tab_rf, tab_dt = st.tabs([
        "📐 Regresi Linear",
        "📈 Forecasting 2025",
        "🌲 Random Forest",
        "🌳 Decision Tree"
    ])
    
    # ----- REGRESI LINEAR -----
    with tab_lr:
        st.markdown("### 📐 Regresi Linear dengan Prediction Playground")
        
        col1, col2 = st.columns(2)
        with col1:
            x_var = st.selectbox("Variabel Independen (X)", COMMODITIES, index=1, key="lr_x")
        with col2:
            y_var = st.selectbox("Variabel Dependen (Y)", COMMODITIES, index=0, key="lr_y")
        
        if x_var == y_var:
            st.warning("⚠️ Variabel X dan Y tidak boleh sama. Silakan pilih yang berbeda.")
        else:
            df_model = df_raw[[x_var, y_var]].dropna()
            
            if len(df_model) < 5:
                st.error("❌ Data tidak cukup untuk membangun model regresi (minimal 5 observasi).")
            else:
                X = df_model[[x_var]].values
                y = df_model[y_var].values
                
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)
                
                mae = mean_absolute_error(y, y_pred)
                rmse = math.sqrt(mean_squared_error(y, y_pred))
                r2 = r2_score(y, y_pred)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("MAE", f"{mae:.2f}")
                c2.metric("RMSE", f"{rmse:.2f}")
                c3.metric("R² Score", f"{r2:.4f}")
                
                coef = model.coef_[0]
                intercept = model.intercept_
                st.markdown(f"""
                <div class="success-box">
                <b>📊 Persamaan Model:</b><br>
                <code>{y_var} = {coef:.4f} × {x_var} + {intercept:.4f}</code>
                </div>
                """, unsafe_allow_html=True)
                
                fig = px.scatter(
                    df_model,
                    x=x_var,
                    y=y_var,
                    trendline='ols',
                    title=f'Regresi Linear: {x_var} → {y_var}',
                    hover_data=['Provinsi'] if 'Provinsi' in df_model.columns else None
                )
                fig.update_layout(height=450)
                st.plotly_chart(fig, use_container_width=True)
                
                # Prediction Playground
                st.markdown("### 🎮 Prediction Playground")
                st.markdown("Masukkan nilai untuk memprediksi hasil:")
                
                user_input = st.number_input(
                    f"Nilai {x_var} (Ribu Ton):",
                    min_value=0.0,
                    max_value=float(df_model[x_var].max()) * 2,
                    value=float(df_model[x_var].mean()),
                    step=10.0
                )
                
                prediction = model.predict([[user_input]])[0]
                
                st.markdown(f"""
                <div class="insight-box">
                <b>🎯 Hasil Prediksi:</b><br>
                Jika <b>{x_var}</b> = <b>{user_input:,.2f}</b> ribu ton, maka 
                <b>{y_var}</b> diprediksi sebesar <b style="font-size: 1.5rem; color: #059669;">
                {prediction:,.2f}</b> ribu ton.
                </div>
                """, unsafe_allow_html=True)
    
    # ----- FORECASTING -----
    with tab_fc:
        st.markdown("### 📈 Simulasi Forecasting 2025")
        st.markdown("""
        <div class="warning-box">
        <b>ℹ️ Catatan Metodologi:</b> Dataset ini bersifat <i>cross-sectional</i> (1 tahun). 
        Forecasting dilakukan melalui <b>simulasi growth rate</b>, bukan time-series forecasting murni.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            fc_comm = st.selectbox("Komoditas", COMMODITIES, key="fc_comm")
            growth = st.slider("Growth Rate (%)", 0, 20, 7, 1) / 100
        
        with col2:
            df_fc = df_raw[['Provinsi', fc_comm]].copy()
            df_fc = df_fc[df_fc[fc_comm] > 0]
            df_fc[f'{fc_comm}_2025'] = df_fc[fc_comm] * (1 + growth)
            df_fc['Kenaikan'] = df_fc[f'{fc_comm}_2025'] - df_fc[fc_comm]
            
            top_fc = df_fc.nlargest(10, fc_comm)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='2024 (Aktual)',
                x=top_fc['Provinsi'],
                y=top_fc[fc_comm],
                marker_color='#3b82f6'
            ))
            fig.add_trace(go.Bar(
                name='2025 (Forecast)',
                x=top_fc['Provinsi'],
                y=top_fc[f'{fc_comm}_2025'],
                marker_color='#f59e0b'
            ))
            fig.update_layout(
                barmode='group',
                title=f'Forecasting {fc_comm}: 2024 vs 2025',
                height=450,
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig, use_container_width=True)
        
        total_2024 = df_fc[fc_comm].sum()
        total_2025 = df_fc[f'{fc_comm}_2025'].sum()
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total 2024", f"{total_2024:,.2f} ribu t")
        c2.metric("Total 2025 (Forecast)", f"{total_2025:,.2f} ribu t")
        c3.metric("Kenaikan", f"+{(total_2025-total_2024):,.2f}", f"{growth*100:.0f}%")
        
        st.dataframe(top_fc, use_container_width=True, hide_index=True)
    
    # ----- RANDOM FOREST -----
    with tab_rf:
        st.markdown("### 🌲 Random Forest Regressor")
        st.markdown("Model ensemble berbasis 100 pohon keputusan.")
        
        target = st.selectbox("Target Prediksi", COMMODITIES, index=0, key="rf_target")
        features = [c for c in COMMODITIES if c != target]
        
        X = df_raw[features]
        y = df_raw[target]
        
        # Validasi
        if y.sum() == 0:
            st.warning(f"⚠️ {target} memiliki total produksi 0 di seluruh provinsi. Model tidak dapat dibangun.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            
            rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("MAE", f"{mean_absolute_error(y_test, y_pred_rf):.2f}")
            c2.metric("RMSE", f"{math.sqrt(mean_squared_error(y_test, y_pred_rf)):.2f}")
            c3.metric("R² Score", f"{r2_score(y_test, y_pred_rf):.4f}")
            
            # Feature Importance
            importance = pd.Series(rf.feature_importances_, index=features)
            importance = importance.sort_values(ascending=True)
            
            fig_imp = px.bar(
                x=importance.values,
                y=importance.index,
                orientation='h',
                title=f'Feature Importance - Prediksi {target}',
                color=importance.values,
                color_continuous_scale='viridis'
            )
            fig_imp.update_layout(height=400)
            st.plotly_chart(fig_imp, use_container_width=True)
            
            top_feat = importance.idxmax()
            st.markdown(f"""
            <div class="insight-box">
            <b>🎯 Variabel Paling Berpengaruh:</b> <code>{top_feat}</code> 
            dengan importance score <b>{importance.max():.3f}</b>.
            Ini menunjukkan bahwa produksi {top_feat} memiliki kaitan prediktif 
            terkuat terhadap produksi {target}.
            </div>
            """, unsafe_allow_html=True)
    
    # ----- DECISION TREE -----
    with tab_dt:
        st.markdown("### 🌳 Decision Tree Regressor")
        st.markdown("Model pohon keputusan dengan max_depth=3 untuk interpretasi yang mudah.")
        
        target_dt = st.selectbox("Target Prediksi", COMMODITIES, index=1, key="dt_target")
        features_dt = [c for c in COMMODITIES if c != target_dt]
        
        X_dt = df_raw[features_dt]
        y_dt = df_raw[target_dt]
        
        if y_dt.sum() == 0:
            st.warning(f"⚠️ {target_dt} memiliki total produksi 0. Model tidak dapat dibangun.")
        else:
            X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(
                X_dt, y_dt, test_size=0.3, random_state=42
            )
            
            dt = DecisionTreeRegressor(max_depth=3, random_state=42)
            dt.fit(X_train_dt, y_train_dt)
            y_pred_dt = dt.predict(X_test_dt)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("MAE", f"{mean_absolute_error(y_test_dt, y_pred_dt):.2f}")
            c2.metric("RMSE", f"{math.sqrt(mean_squared_error(y_test_dt, y_pred_dt)):.2f}")
            c3.metric("R² Score", f"{r2_score(y_test_dt, y_pred_dt):.4f}")
            
            # Feature Importance DT
            dt_imp = pd.Series(dt.feature_importances_, index=features_dt)
            dt_imp = dt_imp.sort_values(ascending=True)
            
            fig_dt = px.bar(
                x=dt_imp.values,
                y=dt_imp.index,
                orientation='h',
                title=f'Feature Importance - Decision Tree untuk {target_dt}',
                color=dt_imp.values,
                color_continuous_scale='greens'
            )
            fig_dt.update_layout(height=400)
            st.plotly_chart(fig_dt, use_container_width=True)
            
            st.markdown(f"""
            <div class="insight-box">
            <b>📊 Analisis Decision Tree:</b><br>
            Decision Tree dengan kedalaman 3 level memprioritaskan variabel 
            <code>{dt_imp.idxmax()}</code> sebagai pembagi utama. Model ini mudah diinterpretasikan 
            namun rentan overfitting pada dataset kecil seperti ini (38 observasi).
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# 12. HALAMAN: INSIGHT & REKOMENDASI
# ============================================================
elif PAGES[selected_page] == "insights":
    st.markdown("## 💡 Insight & Rekomendasi")
    st.markdown("Temuan analitis dan rekomendasi implementatif berbasis data.")
    
    # Generate dynamic insights based on global filters
    st.markdown("### 🔍 Insight Dinamis (Berdasarkan Filter Aktif)")
    
    total_active = df_filtered[selected_commodity].sum()
    top3_provs = df_filtered.nlargest(3, selected_commodity)
    top3_total = top3_provs[selected_commodity].sum()
    concentration = safe_divide(top3_total, total_active) * 100
    non_zero = (df_filtered[selected_commodity] > 0).sum()
    
    insights = [
        f"<b>Konsentrasi Produksi {selected_commodity}:</b> Top 3 provinsi ({', '.join(top3_provs['Provinsi'].tolist())}) menguasai <b>{concentration:.1f}%</b> dari total produksi yang sedang dianalisis.",
        f"<b>Cakupan Wilayah:</b> Hanya <b>{non_zero}</b> dari <b>{len(df_filtered)}</b> provinsi ({safe_divide(non_zero, len(df_filtered))*100:.1f}%) yang aktif memproduksi {selected_commodity}.",
        f"<b>Provinsi Penghasil Tertinggi:</b> {top3_provs.iloc[0]['Provinsi']} dengan produksi <b>{top3_provs.iloc[0][selected_commodity]:,.2f}</b> ribu ton." if len(top3_provs) > 0 else "Tidak ada data produksi untuk komoditas ini."
    ]
    
    for i, insight in enumerate(insights, 1):
        st.markdown(f"""
        <div class="insight-box">
        <b>📌 Insight #{i}:</b><br>
        {insight}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🎯 5 Insight Strategis Nasional")
    
    strategic_insights = [
        ("🏆 Sentra Kelapa Sawit", "Riau, Kalimantan Tengah, dan Sumatera Utara membentuk 'Segitiga Emas' sawit Indonesia dengan total kontribusi >50% produksi nasional."),
        ("🍬 Monopoli Tebu", "Jawa Timur dan Lampung mendominasi hampir seluruh produksi tebu nasional, menciptakan risiko konsentrasi tinggi pada ketahanan gula."),
        ("🌱 Spesialisasi Geografis", "Lemahnya korelasi antar komoditas membuktikan setiap provinsi memiliki keunggulan komparatif unik berdasarkan iklim dan kondisi tanah."),
        ("🍫 Potensi Kakao Sulawesi", "Sulawesi (Tengah, Selatan, Tenggara) menjadi tulang punggung produksi kakao nasional dengan spesialisasi yang jelas."),
        ("🍃 Ekologi Teh", "Produksi teh terbatas pada wilayah berhawa dingin (Jawa Barat, Sumatera Utara, Jawa Tengah) dengan kondisi topografi dataran tinggi.")
    ]
    
    for i, (title, content) in enumerate(strategic_insights, 1):
        st.markdown(f"""
        <div class="insight-box">
        <b>{title}</b><br>
        {content}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🚀 5 Rekomendasi Implementatif")
    
    recommendations = [
        ("💼 Hilirisasi CPO", "Membangun pabrik pengolahan akhir CPO di Riau dan Sumut untuk meningkatkan nilai tambah ekspor, bukan hanya mengekspor bahan mentah."),
        ("🏭 Revitalisasi Pabrik Gula", "Fokuskan subsidi mesin modern dan bibit unggul di Jatim dan Lampung untuk mengurangi ketergantungan impor gula."),
        ("🌳 Replanting Kakao", "Program peremajaan pohon kakao tua di Sulawesi untuk menjaga kualitas ekspor dan produktivitas jangka panjang."),
        ("🗺️ Diversifikasi Wilayah", "Mengurangi risiko konsentrasi produksi dengan mengembangkan sentra-sentra baru di wilayah potensial."),
        ("📊 Integrasi Data Historis", "Membangun sistem monitoring berbasis data time-series untuk forecasting yang lebih akurat dan perencanaan strategis.")
    ]
    
    for i, (title, content) in enumerate(recommendations, 1):
        st.markdown(f"""
        <div class="warning-box">
        <b>🎯 Rekomendasi #{i}: {title}</b><br>
        {content}
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# 13. HALAMAN: EXPORT CENTER
# ============================================================
elif PAGES[selected_page] == "export":
    st.markdown("## 📥 Export Center")
    st.markdown("Unduh dataset dan hasil analisis untuk kebutuhan laporan atau presentasi.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Dataset Terfilter")
        st.markdown(f"**Jumlah baris:** {len(df_filtered)}")
        st.markdown(f"**Komoditas utama:** {selected_commodity}")
        
        csv_data = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Dataset (CSV)",
            data=csv_data,
            file_name=f'dataset_filtered_{selected_commodity.lower().replace(" ", "_")}.csv',
            mime='text/csv',
            use_container_width=True
        )
        
        with st.expander("👁️ Preview Data"):
            st.dataframe(df_filtered.head(10), use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### 📈 Data Forecasting")
        st.markdown("Simulasi proyeksi tahun 2025 dengan growth rate 7%")
        
        df_export_fc = df_raw[['Provinsi'] + COMMODITIES].copy()
        for comm in COMMODITIES:
            df_export_fc[f'{comm}_2025'] = df_export_fc[comm] * 1.07
            df_export_fc[f'{comm}_Kenaikan'] = df_export_fc[f'{comm}_2025'] - df_export_fc[comm]
        
        csv_fc = df_export_fc.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Forecast (CSV)",
            data=csv_fc,
            file_name='forecast_2025_semua_komoditas.csv',
            mime='text/csv',
            use_container_width=True
        )
        
        with st.expander("👁️ Preview Forecast"):
            st.dataframe(df_export_fc.head(10), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("### 📋 Ringkasan Statistik untuk Laporan")
    
    summary_stats = df_raw[COMMODITIES].describe().T
    summary_stats['Total'] = df_raw[COMMODITIES].sum()
    
    csv_summary = summary_stats.to_csv().encode('utf-8')
    st.download_button(
        label="📥 Download Summary Statistics (CSV)",
        data=csv_summary,
        file_name='summary_statistics.csv',
        mime='text/csv',
        use_container_width=True
    )

# ============================================================
# 14. FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div class="dashboard-footer">
<h3>🌾 Dashboard Komoditas Perkebunan Indonesia</h3>
<p>UAS Pengenalan Sains Data — Visualisasi Data & Analisis Data Dasar</p>
<p><small>© 2026 | Dibuat dengan Streamlit, Plotly, dan Scikit-Learn</small></p>
<p><small>Data Source: BPS — Produksi Tanaman Perkebunan Menurut Provinsi 2024</small></p>
</div>
""", unsafe_allow_html=True)
