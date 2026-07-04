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
# DARK PREMIUM CSS
# =========================================================
st.markdown("""
<style>
    /* =========================================================
       GLOBAL
    ========================================================= */
    .stApp {
        background: linear-gradient(180deg, #0b1220 0%, #111827 45%, #0f172a 100%);
        color: #e5e7eb;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    /* =========================================================
       HERO
    ========================================================= */
    .hero-wrap {
        background: linear-gradient(135deg, #111827 0%, #1e3a8a 45%, #0f766e 100%);
        padding: 1.8rem 2rem;
        border-radius: 24px;
        color: #ffffff;
        box-shadow: 0 16px 40px rgba(0,0,0,0.35);
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1.2rem;
    }

    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        line-height: 1.2;
        margin-bottom: 0.35rem;
        color: #ffffff;
    }

    .hero-subtitle {
        font-size: 1rem;
        line-height: 1.6;
        color: rgba(255,255,255,0.88);
        margin-top: 0.35rem;
    }

    .hero-badge {
        display: inline-block;
        padding: 0.42rem 0.9rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.12);
        font-size: 0.9rem;
        color: #f8fafc;
        margin-right: 0.5rem;
        margin-top: 0.7rem;
    }

    /* =========================================================
       SECTION TITLE
    ========================================================= */
    .section-title {
        font-size: 1.28rem;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 0.35rem;
        margin-bottom: 0.9rem;
    }

    .subtle-text {
        color: #cbd5e1;
        font-size: 0.96rem;
        line-height: 1.6;
    }

    /* =========================================================
       CARD / INFO BOX
    ========================================================= */
    .premium-card {
        background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: 0 10px 26px rgba(0,0,0,0.28);
    }

    .info-box {
        background: linear-gradient(135deg, rgba(37,99,235,0.18) 0%, rgba(59,130,246,0.12) 100%);
        border-left: 5px solid #60a5fa;
        padding: 1rem 1rem;
        border-radius: 14px;
        color: #dbeafe;
        margin: 0.5rem 0 0.9rem 0;
        box-shadow: 0 6px 18px rgba(37,99,235,0.10);
    }

    .success-box {
        background: linear-gradient(135deg, rgba(16,185,129,0.16) 0%, rgba(34,197,94,0.10) 100%);
        border-left: 5px solid #34d399;
        padding: 1rem 1rem;
        border-radius: 14px;
        color: #d1fae5;
        margin: 0.5rem 0 0.9rem 0;
        box-shadow: 0 6px 18px rgba(16,185,129,0.10);
    }

    .warn-box {
        background: linear-gradient(135deg, rgba(245,158,11,0.16) 0%, rgba(251,191,36,0.10) 100%);
        border-left: 5px solid #fbbf24;
        padding: 1rem 1rem;
        border-radius: 14px;
        color: #fde68a;
        margin: 0.5rem 0 0.9rem 0;
        box-shadow: 0 6px 18px rgba(245,158,11,0.10);
    }

    .footer-box {
        text-align: center;
        padding: 1.35rem;
        background: linear-gradient(135deg, #111827 0%, #1e40af 100%);
        color: #ffffff;
        border-radius: 18px;
        margin-top: 1.25rem;
        box-shadow: 0 12px 28px rgba(0,0,0,0.25);
        border: 1px solid rgba(255,255,255,0.08);
    }

    /* =========================================================
       SIDEBAR
    ========================================================= */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1220 0%, #111827 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    section[data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }

    /* =========================================================
       METRIC CARD
    ========================================================= */
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 1rem 1rem;
        box-shadow: 0 10px 24px rgba(0,0,0,0.25);
    }

    div[data-testid="stMetricLabel"] {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-weight: 800 !important;
    }

    div[data-testid="stMetricDelta"] {
        color: #34d399 !important;
    }

    /* =========================================================
       TABS
    ========================================================= */
    button[data-baseweb="tab"] {
        border-radius: 10px !important;
        font-weight: 700 !important;
        background: #111827 !important;
        color: #e5e7eb !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #1d4ed8 0%, #0f766e 100%) !important;
        color: white !important;
        border: none !important;
    }

    /* =========================================================
       DATAFRAME / TABLE / EXPANDER
    ========================================================= */
    .stDataFrame, .stTable {
        border-radius: 14px;
        overflow: hidden;
    }

    div[data-testid="stExpander"] {
        background: #111827;
        border: 1px solid rgba(255,255,255,0.06);
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
        color: #f8fafc !important;
        font-weight: 600;
    }

    /* =========================================================
       HIDE STREAMLIT SMALL HEADER SPACE
    ========================================================= */
    header[data-testid="stHeader"] {
        background: rgba(0,0,0,0);
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
# HELPER
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
# PLOTLY DARK TEMPLATE
# =========================================================
plot_bg = "#111827"
paper_bg = "#111827"
font_color = "#f8fafc"
grid_color = "rgba(255,255,255,0.08)"

def apply_dark_layout(fig, height=500):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(color=font_color),
        height=height,
        margin=dict(l=40, r=30, t=60, b=40),
        xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0.08)"
        )
    )
    return fig

# =========================================================
# HERO HEADER
# =========================================================
st.markdown("""
<div class="hero-wrap">
    <div class="hero-title">🌾 Premium Dashboard Komoditas Perkebunan Indonesia</div>
    <div class="hero-subtitle">
        Dashboard analitik interaktif untuk eksplorasi, visualisasi, dan pemodelan data produksi komoditas perkebunan Indonesia per provinsi (2024). 
        Dirancang dengan pendekatan executive dashboard, EDA interaktif, predictive analytics, dan insight generator yang lebih presentable untuk konteks akademik maupun demonstrasi proyek.
    </div>
    <div>
        <span class="hero-badge">Executive Dashboard</span>
        <span class="hero-badge">Interactive EDA</span>
        <span class="hero-badge">Machine Learning Playground</span>
        <span class="hero-badge">Export Ready</span>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("🧭 Dashboard Navigation")

menu = st.sidebar.radio(
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
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Global Filter")

selected_commodity = st.sidebar.selectbox("Komoditas utama", numeric_cols, index=0)
province_options = ["Semua Provinsi"] + df["Provinsi"].tolist()
selected_province = st.sidebar.selectbox("Provinsi", province_options, index=0)
top_n = st.sidebar.slider("Top N Provinsi", min_value=5, max_value=20, value=10)
show_zero_rows = st.sidebar.checkbox("Tampilkan provinsi dengan total produksi 0", value=True)

view_mode = st.sidebar.radio(
    "Mode tampilan",
    ["Nilai Absolut", "Persentase (%)"],
    index=0
)

filtered_df = get_filtered_df(df, selected_province, show_zero_rows)

# =========================================================
# EXECUTIVE DASHBOARD
# =========================================================
if menu == "🏠 Executive Dashboard":
    st.markdown('<div class="section-title">Executive Dashboard</div>', unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("Tidak ada data yang cocok dengan filter aktif.")
    else:
        total_production = filtered_df[numeric_cols].sum().sum()
        total_province = filtered_df.shape[0]
        commodity_totals = filtered_df[numeric_cols].sum().sort_values(ascending=False)
        top_commodity = commodity_totals.index[0]
        top_commodity_val = commodity_totals.iloc[0]
        best_prov, best_total = province_with_highest_total(filtered_df)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Jumlah Provinsi", total_province)
        c2.metric("Total Produksi", format_num(total_production))
        c3.metric("Komoditas Dominan", top_commodity, format_num(top_commodity_val))
        c4.metric("Provinsi Tertinggi", best_prov, format_num(best_total))

        st.markdown("")

        left, right = st.columns([1.2, 1])

        with left:
            st.markdown('<div class="section-title">Top Provinsi berdasarkan Komoditas Terpilih</div>', unsafe_allow_html=True)
            top_df = filtered_df[["Provinsi", selected_commodity]].sort_values(selected_commodity, ascending=False).head(min(top_n, len(filtered_df))).copy()

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
            fig = apply_dark_layout(fig, 540)
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.markdown('<div class="section-title">Komposisi Produksi Antar Komoditas</div>', unsafe_allow_html=True)
            comp_df = filtered_df[numeric_cols].sum().reset_index()
            comp_df.columns = ["Komoditas", "Produksi"]
            if view_mode == "Persentase (%)":
                total_comp = comp_df["Produksi"].sum()
                comp_df["Produksi"] = (comp_df["Produksi"] / total_comp * 100) if total_comp > 0 else 0

            fig2 = px.pie(
                comp_df,
                values="Produksi",
                names="Komoditas",
                hole=0.5
            )
            fig2.update_traces(textinfo='percent+label')
            fig2 = apply_dark_layout(fig2, 540)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-title">Quick Insight</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="info-box">{generate_dynamic_insight(filtered_df, selected_commodity)}</div>',
            unsafe_allow_html=True
        )

        st.markdown('<div class="section-title">Ringkasan Produksi per Provinsi</div>', unsafe_allow_html=True)
        summary_df = add_total_production(filtered_df)
        st.dataframe(summary_df, use_container_width=True)

# =========================================================
# DATA EXPLORER
# =========================================================
elif menu == "📊 Data Explorer":
    st.markdown('<div class="section-title">Data Explorer</div>', unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("Tidak ada data untuk filter aktif.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Jumlah Observasi", filtered_df.shape[0])
        c2.metric("Jumlah Variabel", filtered_df.shape[1])
        c3.metric("Missing Value", int(filtered_df.isnull().sum().sum()))

        tab1, tab2, tab3 = st.tabs(["📋 Dataset", "📈 Statistik Deskriptif", "🧹 Kualitas Data"])

        with tab1:
            st.dataframe(filtered_df, use_container_width=True)

        with tab2:
            desc = filtered_df[numeric_cols].describe().T
            desc["range"] = desc["max"] - desc["min"]
            st.dataframe(desc, use_container_width=True)

        with tab3:
            duplicate_count = filtered_df.duplicated().sum()
            zero_rows = (filtered_df[numeric_cols].sum(axis=1) == 0).sum()

            st.markdown(f"- **Missing values:** {int(filtered_df.isnull().sum().sum())}")
            st.markdown(f"- **Baris duplikat:** {int(duplicate_count)}")
            st.markdown(f"- **Provinsi dengan total produksi 0:** {int(zero_rows)}")
            st.markdown(
                '<div class="warn-box">Outlier tidak otomatis dihapus karena bisa merepresentasikan sentra produksi riil, bukan kesalahan data.</div>',
                unsafe_allow_html=True
            )

# =========================================================
# EDA EXPLORER
# =========================================================
elif menu == "🔍 EDA Explorer":
    st.markdown('<div class="section-title">EDA Explorer</div>', unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("Tidak ada data untuk filter aktif.")
    else:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📌 Overview", "📊 Distribution", "🔗 Relationship", "🔥 Correlation", "📍 Province Deep Dive"
        ])

        with tab1:
            c1, c2 = st.columns([1.2, 1])

            with c1:
                top_df = filtered_df[["Provinsi", selected_commodity]].sort_values(selected_commodity, ascending=False).head(min(top_n, len(filtered_df))).copy()
                fig = px.bar(
                    top_df.sort_values(selected_commodity, ascending=True),
                    x=selected_commodity,
                    y="Provinsi",
                    orientation="h",
                    text=selected_commodity,
                    title=f"Top {min(top_n, len(top_df))} Provinsi - {selected_commodity}"
                )
                fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                fig = apply_dark_layout(fig, 500)
                st.plotly_chart(fig, use_container_width=True)

            with c2:
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
                fig2 = apply_dark_layout(fig2, 500)
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown(
                f'<div class="info-box">{generate_dynamic_insight(filtered_df, selected_commodity)}</div>',
                unsafe_allow_html=True
            )

        with tab2:
            c1, c2 = st.columns(2)

            with c1:
                fig = px.histogram(
                    filtered_df,
                    x=selected_commodity,
                    nbins=15,
                    title=f"Distribusi {selected_commodity}"
                )
                fig = apply_dark_layout(fig, 480)
                st.plotly_chart(fig, use_container_width=True)

            with c2:
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
                fig2 = apply_dark_layout(fig2, 480)
                st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            x_var = st.selectbox("Variabel X", numeric_cols, index=0, key="eda_x")
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
                fig = apply_dark_layout(fig, 550)
                st.plotly_chart(fig, use_container_width=True)

                corr_val = filtered_df[[x_var, y_var]].corr().iloc[0, 1]
                if pd.notna(corr_val):
                    st.markdown(
                        f'<div class="info-box">Nilai korelasi Pearson antara <b>{x_var}</b> dan <b>{y_var}</b> adalah <b>{corr_val:.3f}</b>.</div>',
                        unsafe_allow_html=True
                    )

with tab4:
    corr = filtered_df[numeric_cols].corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Heatmap Korelasi Antar Komoditas"
    )

    fig.update_traces(
        textfont=dict(color="white", size=12),
        xgap=2,
        ygap=2
    )

    fig.update_xaxes(side="bottom", tickangle=30)
    fig.update_yaxes(autorange="reversed")

    fig.update_layout(
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(color="#f8fafc"),
        margin=dict(l=40, r=30, t=70, b=40),
        coloraxis_colorbar=dict(
            title="Korelasi",
            tickfont=dict(color="#f8fafc"),
            titlefont=dict(color="#f8fafc")
        )
    )

    st.plotly_chart(fig, use_container_width=True)

        with tab5:
            province_pick = st.selectbox("Pilih provinsi", filtered_df["Provinsi"].tolist(), key="deep_prov")
            p_df = filtered_df[filtered_df["Provinsi"] == province_pick]

            if not p_df.empty:
                row = p_df.iloc[0]
                profile = pd.DataFrame({
                    "Komoditas": numeric_cols,
                    "Produksi": [row[c] for c in numeric_cols]
                }).sort_values("Produksi", ascending=False)

                c1, c2 = st.columns([1.1, 1])
                with c1:
                    fig = px.bar(profile, x="Komoditas", y="Produksi", text="Produksi", title=f"Profil Produksi {province_pick}")
                    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                    fig = apply_dark_layout(fig, 500)
                    st.plotly_chart(fig, use_container_width=True)

                with c2:
                    total_p = profile["Produksi"].sum()
                    dom_comm = profile.iloc[0]["Komoditas"]
                    dom_val = profile.iloc[0]["Produksi"]

                    st.metric("Total Produksi Provinsi", format_num(total_p))
                    st.metric("Komoditas Dominan", dom_comm, format_num(dom_val))

                    fig2 = px.pie(profile, names="Komoditas", values="Produksi", hole=0.45, title="Komposisi Komoditas")
                    fig2.update_traces(textinfo='percent+label')
                    fig2 = apply_dark_layout(fig2, 420)
                    st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# PROFIL PROVINSI & KOMODITAS
# =========================================================
elif menu == "🧭 Profil Provinsi & Komoditas":
    st.markdown('<div class="section-title">Profil Provinsi & Komoditas</div>', unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("Tidak ada data untuk filter aktif.")
    else:
        tab_p, tab_k = st.tabs(["📍 Profil Provinsi", "🌾 Profil Komoditas"])

        with tab_p:
            province_pick = st.selectbox("Pilih provinsi", filtered_df["Provinsi"].tolist(), key="profile_province")
            p_df = filtered_df[filtered_df["Provinsi"] == province_pick]

            if not p_df.empty:
                row = p_df.iloc[0]
                prov_profile = pd.DataFrame({
                    "Komoditas": numeric_cols,
                    "Produksi": [row[c] for c in numeric_cols]
                }).sort_values("Produksi", ascending=False)

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Produksi", format_num(prov_profile["Produksi"].sum()))
                c2.metric("Komoditas Dominan", prov_profile.iloc[0]["Komoditas"])
                c3.metric("Nilai Tertinggi", format_num(prov_profile.iloc[0]["Produksi"]))

                fig = px.bar(
                    prov_profile,
                    x="Komoditas",
                    y="Produksi",
                    text="Produksi",
                    title=f"Struktur Komoditas - {province_pick}"
                )
                fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                fig = apply_dark_layout(fig, 520)
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(prov_profile, use_container_width=True)

        with tab_k:
            commodity_pick = st.selectbox("Pilih komoditas", numeric_cols, key="profile_commodity")
            cdf = filtered_df[["Provinsi", commodity_pick]].sort_values(commodity_pick, ascending=False).copy()
            total = cdf[commodity_pick].sum()

            top_prov, top_val = top_province_for_commodity(filtered_df, commodity_pick)

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Nasional", format_num(total))
            c2.metric("Provinsi Tertinggi", top_prov)
            c3.metric("Produksi Tertinggi", format_num(top_val))

            fig = px.bar(
                cdf.head(min(top_n, len(cdf))),
                x="Provinsi",
                y=commodity_pick,
                text=commodity_pick,
                title=f"Top {min(top_n, len(cdf))} Provinsi - {commodity_pick}"
            )
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_xaxes(tickangle=45)
            fig = apply_dark_layout(fig, 520)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                f'<div class="info-box">{generate_dynamic_insight(filtered_df, commodity_pick)}</div>',
                unsafe_allow_html=True
            )

# =========================================================
# GEO INSIGHT
# =========================================================
elif menu == "🗺️ Geo Insight":
    st.markdown('<div class="section-title">Geo Insight</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtle-text">Halaman ini menampilkan persebaran produksi komoditas terpilih dalam bentuk ranking spasial dan bubble geo-insight. Jika nanti kamu ingin, halaman ini bisa saya upgrade ke peta Indonesia provinsi berbasis GeoJSON.</div>',
        unsafe_allow_html=True
    )

    if filtered_df.empty:
        st.warning("Tidak ada data untuk filter aktif.")
    else:
        geo_df = filtered_df[["Provinsi", selected_commodity]].copy().sort_values(selected_commodity, ascending=False)

        c1, c2 = st.columns([1.2, 1])

        with c1:
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
            fig = apply_dark_layout(fig, 520)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            pseudo_geo = geo_df.head(min(15, len(geo_df))).copy()
            pseudo_geo["Rank"] = range(1, len(pseudo_geo) + 1)

            fig2 = px.scatter(
                pseudo_geo,
                x="Rank",
                y=selected_commodity,
                size=selected_commodity,
                hover_name="Provinsi",
                text="Provinsi",
                title="Bubble Geo-Insight (Simulasi Sebaran Prioritas)"
            )
            fig2.update_traces(textposition="top center")
            fig2 = apply_dark_layout(fig2, 520)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown(
            f'<div class="info-box">Komoditas <b>{selected_commodity}</b> memperlihatkan konsentrasi produksi pada sejumlah provinsi utama. Versi premium berikutnya bisa ditingkatkan menjadi <b>choropleth map Indonesia</b> berbasis GeoJSON agar narasi geografis lebih kuat.</div>',
            unsafe_allow_html=True
        )

# =========================================================
# PREDICTIVE ANALYTICS
# =========================================================
elif menu == "📈 Predictive Analytics":
    st.markdown('<div class="section-title">Predictive Analytics</div>', unsafe_allow_html=True)

    if filtered_df.shape[0] < 5:
        st.warning("Data terlalu sedikit untuk analisis prediktif yang stabil.")
    else:
        tab_lr, tab_fc, tab_rf, tab_dt = st.tabs([
            "📉 Regresi Linear", "📈 Forecasting", "🌲 Random Forest", "🌳 Decision Tree"
        ])

        # -------------------------------------------------
        # REGRESI LINEAR
        # -------------------------------------------------
        with tab_lr:
            st.markdown("### Regresi Linear Interaktif")

            c1, c2 = st.columns(2)
            with c1:
                x_var = st.selectbox("Variabel independen (X)", numeric_cols, index=1 if len(numeric_cols) > 1 else 0, key="lr_x")
            with c2:
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

                    k1, k2, k3 = st.columns(3)
                    k1.metric("MAE", f"{mae:.2f}")
                    k2.metric("RMSE", f"{rmse:.2f}")
                    k3.metric("R²", f"{r2:.4f}")

                    st.markdown(
                        f'<div class="info-box"><b>Persamaan model:</b> {y_var} = {lr.coef_[0]:.4f} × {x_var} + {lr.intercept_:.4f}</div>',
                        unsafe_allow_html=True
                    )

                    plot_df = model_df.copy()
                    plot_df["Prediksi"] = y_pred

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=plot_df[x_var],
                        y=plot_df[y_var],
                        mode="markers",
                        name="Aktual"
                    ))

                    sort_idx = np.argsort(plot_df[x_var].values)
                    fig.add_trace(go.Scatter(
                        x=plot_df[x_var].values[sort_idx],
                        y=plot_df["Prediksi"].values[sort_idx],
                        mode="lines",
                        name="Garis Regresi"
                    ))

                    fig.update_layout(
                        title=f"Regresi Linear: {x_var} vs {y_var}",
                        xaxis_title=x_var,
                        yaxis_title=y_var
                    )
                    fig = apply_dark_layout(fig, 540)
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("### Prediction Playground")
                    x_input = st.number_input(
                        f"Masukkan nilai {x_var} untuk memprediksi {y_var}",
                        min_value=0.0,
                        value=float(np.median(model_df[x_var]))
                    )
                    pred_val = lr.predict(np.array([[x_input]]))[0]
                    st.success(f"Prediksi **{y_var}** untuk nilai **{x_var} = {x_input:.2f}** adalah **{pred_val:.2f}**.")

        # -------------------------------------------------
        # FORECASTING
        # -------------------------------------------------
        with tab_fc:
            st.markdown("### Forecasting 2025 (Simulasi Growth Rate)")
            st.markdown(
                '<div class="warn-box">Dataset bersifat cross-sectional (1 tahun), sehingga forecasting di dashboard ini diposisikan sebagai simulasi pertumbuhan, bukan time-series forecasting historis.</div>',
                unsafe_allow_html=True
            )

            commodity_target = st.selectbox("Pilih komoditas untuk forecast", numeric_cols, key="fc_target")
            growth_rate = st.slider("Growth Rate (%)", min_value=1, max_value=20, value=7, key="fc_growth") / 100

            fc_df = filtered_df[["Provinsi", commodity_target]].copy()
            fc_df["Forecast_2025"] = fc_df[commodity_target] * (1 + growth_rate)
            fc_df["Peningkatan"] = fc_df["Forecast_2025"] - fc_df[commodity_target]

            top_fc = fc_df.sort_values(commodity_target, ascending=False).head(min(top_n, len(fc_df))).copy()

            fig = go.Figure()
            fig.add_trace(go.Bar(x=top_fc["Provinsi"], y=top_fc[commodity_target], name="2024"))
            fig.add_trace(go.Bar(x=top_fc["Provinsi"], y=top_fc["Forecast_2025"], name="2025 (Forecast)"))
            fig.update_layout(
                barmode="group",
                title=f"Perbandingan {commodity_target}: 2024 vs 2025",
                xaxis_title="Provinsi",
                yaxis_title="Produksi"
            )
            fig = apply_dark_layout(fig, 550)
            st.plotly_chart(fig, use_container_width=True)

            total_now = fc_df[commodity_target].sum()
            total_fc = fc_df["Forecast_2025"].sum()

            k1, k2, k3 = st.columns(3)
            k1.metric("Total 2024", format_num(total_now))
            k2.metric("Total 2025 (Forecast)", format_num(total_fc))
            k3.metric("Kenaikan Total", format_num(total_fc - total_now))

            st.dataframe(fc_df.sort_values("Forecast_2025", ascending=False), use_container_width=True)

        # -------------------------------------------------
        # RANDOM FOREST
        # -------------------------------------------------
        with tab_rf:
            st.markdown("### Random Forest Regression")
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

                k1, k2, k3 = st.columns(3)
                k1.metric("MAE", f"{mae:.2f}")
                k2.metric("RMSE", f"{rmse:.2f}")
                k3.metric("R²", f"{r2:.4f}")

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
                fig = apply_dark_layout(fig, 500)
                st.plotly_chart(fig, use_container_width=True)

                compare_df = pd.DataFrame({
                    "Aktual": y_test.values,
                    "Prediksi": y_pred
                })
                st.dataframe(compare_df, use_container_width=True)

        # -------------------------------------------------
        # DECISION TREE
        # -------------------------------------------------
        with tab_dt:
            st.markdown("### Decision Tree Regression")
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

                k1, k2, k3 = st.columns(3)
                k1.metric("MAE", f"{mae:.2f}")
                k2.metric("RMSE", f"{rmse:.2f}")
                k3.metric("R²", f"{r2:.4f}")

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
                fig = apply_dark_layout(fig, 500)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown(
                    '<div class="info-box">Visualisasi pohon keputusan penuh tidak ditampilkan agar dashboard tetap ringan dan stabil. Interpretasi model difokuskan pada <b>feature importance</b>.</div>',
                    unsafe_allow_html=True
                )

# =========================================================
# INSIGHT & REKOMENDASI
# =========================================================
elif menu == "🧠 Insight & Rekomendasi":
    st.markdown('<div class="section-title">Insight & Rekomendasi</div>', unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("Tidak ada data untuk filter aktif.")
    else:
        total_by_comm = filtered_df[numeric_cols].sum().sort_values(ascending=False)
        dominant_comm = total_by_comm.index[0]
        dominant_val = total_by_comm.iloc[0]

        top_prov, top_val = top_province_for_commodity(filtered_df, selected_commodity)
        total_selected = filtered_df[selected_commodity].sum()
        share_top = (top_val / total_selected * 100) if total_selected > 0 else 0

        insights = [
            f"Komoditas dengan total produksi terbesar pada filter aktif adalah <b>{dominant_comm}</b> sebesar <b>{format_num(dominant_val)} ribu ton</b>.",
            f"Untuk komoditas <b>{selected_commodity}</b>, provinsi tertinggi adalah <b>{top_prov}</b> dengan produksi <b>{format_num(top_val)} ribu ton</b>.",
            f"Kontribusi <b>{top_prov}</b> terhadap total <b>{selected_commodity}</b> mencapai sekitar <b>{share_top:.1f}%</b>.",
            "Korelasi antar komoditas perlu dibaca hati-hati karena produksi perkebunan dipengaruhi spesialisasi wilayah, kondisi agroklimat, dan struktur ekonomi regional.",
            "Forecasting pada dashboard ini bersifat simulatif sehingga lebih tepat digunakan untuk eksplorasi skenario awal daripada prediksi kebijakan final."
        ]

        st.markdown("### 💡 Insight Utama")
        for i, ins in enumerate(insights, start=1):
            st.markdown(f'<div class="info-box"><b>{i}.</b> {ins}</div>', unsafe_allow_html=True)

        st.markdown("### 🚀 Rekomendasi Strategis")
        recs = generate_recommendations(filtered_df, selected_commodity)
        for i, rec in enumerate(recs, start=1):
            st.markdown(f'<div class="success-box"><b>{i}.</b> {rec}</div>', unsafe_allow_html=True)

# =========================================================
# EXPORT CENTER
# =========================================================
elif menu == "📥 Export Center":
    st.markdown('<div class="section-title">Export Center</div>', unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("Tidak ada data untuk diekspor.")
    else:
        st.markdown("Unduh data hasil filter aktif atau hasil forecast untuk kebutuhan pelaporan dan presentasi.")

        export_df = add_total_production(filtered_df)
        st.download_button(
            label="⬇️ Download dataset hasil filter (CSV)",
            data=export_csv(export_df),
            file_name="dataset_hasil_filter.csv",
            mime="text/csv"
        )

        st.markdown("### Export Forecast")
        exp_comm = st.selectbox("Komoditas untuk file forecast", numeric_cols, key="exp_comm")
        exp_growth = st.slider("Growth rate forecast (%)", 1, 20, 7, key="exp_growth") / 100

        export_fc = filtered_df[["Provinsi", exp_comm]].copy()
        export_fc["Forecast_2025"] = export_fc[exp_comm] * (1 + exp_growth)
        export_fc["Peningkatan"] = export_fc["Forecast_2025"] - export_fc[exp_comm]

        st.dataframe(export_fc, use_container_width=True)

        st.download_button(
            label="⬇️ Download forecast 2025 (CSV)",
            data=export_csv(export_fc),
            file_name=f"forecast_{exp_comm.lower().replace(' ', '_')}_2025.csv",
            mime="text/csv"
        )

# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="footer-box">
    <h3>🎓 Premium Dashboard — UAS Pengenalan Sains Data</h3>
    <p>Versi dark theme dengan executive dashboard, EDA interaktif, predictive analytics, dan export center.</p>
</div>
""", unsafe_allow_html=True)
