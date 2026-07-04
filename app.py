import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Indonesian Plantation Intelligence Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# ULTRA PREMIUM DARK THEME CSS
# =========================================================
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #0b1220;
        color: #f8fafc;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .block-container {
        max-width: 1600px;
        padding: 1rem 2rem 3rem 2rem;
    }
    
    /* Custom KPI Cards */
    .custom-kpi-card {
        background: linear-gradient(145deg, #111827, #0b1220);
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        border-left: 5px solid #3b82f6;
        transition: all 0.3s ease;
    }
    
    .custom-kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 15px 40px rgba(59, 130, 246, 0.3);
    }
    
    .kpi-title {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #f8fafc;
        line-height: 1.2;
        letter-spacing: -0.02em;
    }
    
    .kpi-subtitle {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 0.25rem;
        font-weight: 500;
    }
    
    /* Info Boxes */
    .premium-info-box {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(99, 102, 241, 0.05) 100%);
        border-left: 3px solid #3b82f6;
        padding: 1rem 1.25rem;
        border-radius: 12px;
        color: #e2e8f0;
        line-height: 1.6;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.1);
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(251, 191, 36, 0.05) 100%);
        border-left: 3px solid #f59e0b;
        padding: 1rem 1.25rem;
        border-radius: 12px;
        color: #fde68a;
        line-height: 1.6;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(52, 211, 153, 0.05) 100%);
        border-left: 3px solid #10b981;
        padding: 1rem 1.25rem;
        border-radius: 12px;
        color: #d1fae5;
        line-height: 1.6;
        margin: 1rem 0;
    }
    
    /* Sidebar Styles */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #0b1220 100%);
        border-right: 1px solid rgba(59, 130, 246, 0.2);
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown {
        color: #cbd5e1 !important;
    }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #111827, #0b1220);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 12px;
        padding: 1rem;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
    }
    
    div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-size: 1.8rem !important;
        font-weight: 800 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.4);
    }
    
    /* DataFrames */
    .stDataFrame {
        border-radius: 12px;
        border: 1px solid rgba(59, 130, 246, 0.2) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(17, 24, 39, 0.6);
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
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%) !important;
        color: white !important;
    }
    
    /* Hero Header */
    .hero-container {
        background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 50%, #0f766e 100%);
        border-radius: 24px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(30, 58, 138, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.5rem;
        letter-spacing: -0.03em;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.85);
        line-height: 1.6;
    }
    
    /* Section Headers */
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
    
    /* Filter Section */
    .filter-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# DATASET EMBEDDED (BPS 2024 Data)
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
# HELPER FUNCTIONS
# =========================================================
def format_num(x):
    """Format angka dengan suffix K/M"""
    try:
        val = float(x)
        if val >= 1_000_000:
            return f"{val/1_000_000:.1f}M"
        if val >= 1_000:
            return f"{val/1_000:.1f}K"
        return f"{val:,.0f}"
    except:
        return str(x)

def add_total_production(data):
    """Tambahkan kolom total produksi"""
    temp = data.copy()
    temp["Total Produksi"] = temp[numeric_cols].sum(axis=1)
    return temp

def apply_dark_layout(fig, height=500):
    """Apply dark theme ke Plotly chart"""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(11, 18, 32, 0)",
        plot_bgcolor="rgba(17, 24, 39, 0.4)",
        font=dict(color="#f8fafc", family="Inter, sans-serif"),
        height=height,
        margin=dict(l=40, r=30, t=60, b=40),
        xaxis=dict(gridcolor="rgba(148, 163, 184, 0.1)", zerolinecolor="rgba(148, 163, 184, 0.2)"),
        yaxis=dict(gridcolor="rgba(148, 163, 184, 0.1)", zerolinecolor="rgba(148, 163, 184, 0.2)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(148, 163, 184, 0.2)")
    )
    return fig

def create_kpi_card(title, value, subtitle="", icon="📊", color="#3b82f6"):
    """Buat custom KPI card"""
    return f"""
    <div class="custom-kpi-card" style="border-left-color: {color};">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div class="kpi-title">{title}</div>
            <div style="font-size: 1.5rem;">{icon}</div>
        </div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-subtitle">{subtitle}</div>
    </div>
    """

def generate_insight_box(text, box_type="info"):
    """Generate insight box"""
    if box_type == "warning":
        return f'<div class="warning-box">{text}</div>'
    elif box_type == "success":
        return f'<div class="success-box">{text}</div>'
    return f'<div class="premium-info-box">{text}</div>'

def export_csv(dataframe):
    """Export dataframe ke CSV"""
    return dataframe.to_csv(index=False).encode("utf-8")

# =========================================================
# SIDEBAR CONTROL CENTER
# =========================================================
with st.sidebar:
    st.markdown("## 🎛️ Intelligence Control")
    
    st.markdown('<div class="filter-title">Primary Commodity Focus</div>', unsafe_allow_html=True)
    selected_commodity = st.selectbox(
        "Commodity",
        numeric_cols,
        index=0,
        label_visibility="collapsed",
        key="side_comm_select"
    )
    
    st.markdown('<div class="filter-title">Province Filter</div>', unsafe_allow_html=True)
    prov_options = ["All Provinces"] + df["Provinsi"].tolist()
    selected_province = st.selectbox(
        "Province",
        prov_options,
        index=0,
        label_visibility="collapsed",
        key="side_prov_select"
    )
    
    st.markdown('<div class="filter-title">Top-N Ranking Limit</div>', unsafe_allow_html=True)
    top_n = st.slider(
        "Top N",
        min_value=5,
        max_value=20,
        value=10,
        label_visibility="collapsed",
        key="side_topn_slider"
    )
    
    st.markdown("---")
    st.markdown("## 🧭 Navigation Matrix")
    
    menu = st.radio(
        "Navigation",
        [
            "🏠 Executive Overview",
            "🌾 Commodity Intelligence",
            "🗺️ Province Intelligence",
            "📊 Comparative Analytics",
            "🤖 Predictive Analytics",
            "📦 Data & Export"
        ],
        label_visibility="collapsed",
        key="side_menu_radio"
    )

# Apply Global Filter
active_df = df.copy()
if selected_province != "All Provinces":
    active_df = active_df[active_df["Provinsi"] == selected_province].copy()

# =========================================================
# HERO HEADER (Only on Executive Overview)
# =========================================================
if menu == "🏠 Executive Overview":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🌾 Indonesian Plantation Intelligence Matrix</div>
        <div class="hero-subtitle">
            Executive analytics dashboard for national commodity performance analysis. 
            Powered by advanced data science algorithms and premium data visualization. 
            Data Source: BPS - Produksi Tanaman Perkebunan 2024
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"<h1 style='color: #f8fafc; font-weight: 800; margin-bottom: 0.5rem;'>{menu}</h1>", unsafe_allow_html=True)
    st.markdown("---")

# =========================================================
# PAGE 1: EXECUTIVE OVERVIEW
# =========================================================
if menu == "🏠 Executive Overview":
    
    # Calculate KPIs
    total_nat_prod = active_df[numeric_cols].sum().sum()
    total_provs = len(active_df)
    
    comm_sums = active_df[numeric_cols].sum().sort_values(ascending=False)
    dom_comm = comm_sums.index[0] if not comm_sums.empty else "-"
    dom_val = comm_sums.iloc[0] if not comm_sums.empty else 0
    
    temp_avg = total_nat_prod / max(1, total_provs)
    
    temp_tot = add_total_production(active_df).sort_values("Total Produksi", ascending=False)
    top_prov_name = temp_tot.iloc[0]["Provinsi"] if not temp_tot.empty else "-"
    
    top5_share = temp_tot.head(5)["Total Produksi"].sum() / max(1, temp_tot["Total Produksi"].sum()) * 100
    
    active_provs = (active_df[numeric_cols].sum(axis=1) > 0).sum()

    # Display KPI Cards
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(create_kpi_card("National Production", format_num(total_nat_prod), "Total Ribu Ton", "🇮🇩", "#3b82f6"), unsafe_allow_html=True)
    with c2:
        st.markdown(create_kpi_card("Dominant Commodity", dom_comm, format_num(dom_val), "🏆", "#10b981"), unsafe_allow_html=True)
    with c3:
        st.markdown(create_kpi_card("Top Producer", top_prov_name, "Highest Yield", "📍", "#f59e0b"), unsafe_allow_html=True)
    with c4:
        st.markdown(create_kpi_card("Active Provinces", str(active_provs), f"out of {total_provs}", "🗺️", "#8b5cf6"), unsafe_allow_html=True)
    with c5:
        st.markdown(create_kpi_card("Top-5 Share", f"{top5_share:.1f}%", "Market Concentration", "🎯", "#ec4899"), unsafe_allow_html=True)
    with c6:
        st.markdown(create_kpi_card("Avg per Province", format_num(temp_avg), "Ribu Ton", "📊", "#06b6d4"), unsafe_allow_html=True)

    st.markdown("### 📈 Strategic Rankings")
    col_l, col_r = st.columns(2)
    
    with col_l:
        top_comm_df = active_df[["Provinsi", selected_commodity]].sort_values(selected_commodity, ascending=False).head(top_n)
        fig1 = px.bar(
            top_comm_df[::-1],
            x=selected_commodity,
            y="Provinsi",
            orientation='h',
            text=selected_commodity,
            title=f"Top {top_n} Provinces: {selected_commodity}"
        )
        fig1.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        fig1 = apply_dark_layout(fig1, 550)
        fig1.update_layout(xaxis_title="Production (K Ton)", yaxis_title="")
        st.plotly_chart(fig1, use_container_width=True, key="exec_bar_chart_001")
        
    with col_r:
        comp_df = active_df[numeric_cols].sum().reset_index()
        comp_df.columns = ["Commodity", "Value"]
        fig2 = go.Figure(go.Pie(
            labels=comp_df["Commodity"],
            values=comp_df["Value"],
            hole=0.55,
            marker=dict(colors=px.colors.qualitative.Vivid)
        ))
        fig2.update_traces(textinfo='percent+label')
        fig2 = apply_dark_layout(fig2, 550)
        fig2.update_layout(title="National Composition Matrix")
        st.plotly_chart(fig2, use_container_width=True, key="exec_pie_chart_001")

    st.markdown("### 💡 Automated Insights")
    st.markdown(generate_insight_box(
        f"<b>Market Concentration Warning:</b> The top 5 provinces control <b>{top5_share:.1f}%</b> of total national output. "
        f"Currently, <b>{dom_comm}</b> dominates with <b>{format_num(dom_val)}</b> K Ton. "
        f"Strategic recommendation: Diversify production centers to mitigate regional climate/economic risks."
    ), unsafe_allow_html=True)


# =========================================================
# PAGE 2: COMMODITY INTELLIGENCE
# =========================================================
elif menu == "🌾 Commodity Intelligence":
    target_comm = st.selectbox("Select Commodity to Analyze", numeric_cols, key="comm_int_select")
    
    comm_df = active_df[["Provinsi", target_comm]].copy()
    comm_df = comm_df[comm_df[target_comm] > 0].sort_values(target_comm, ascending=False)
    
    tot_c = comm_df[target_comm].sum()
    top_prov_c = comm_df.iloc[0]["Provinsi"] if not comm_df.empty else "-"
    top_val_c = comm_df.iloc[0][target_comm] if not comm_df.empty else 0
    med_c = comm_df[target_comm].median()
    top5_c_share = comm_df.head(5)[target_comm].sum() / max(1, tot_c) * 100
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(create_kpi_card("Total Yield", format_num(tot_c), "Ribu Ton", "📦", "#3b82f6"), unsafe_allow_html=True)
    with c2:
        st.markdown(create_kpi_card("Market Leader", top_prov_c, format_num(top_val_c), "🥇", "#10b981"), unsafe_allow_html=True)
    with c3:
        st.markdown(create_kpi_card("Median Output", format_num(med_c if not pd.isna(med_c) else 0), "Ribu Ton", "⚖️", "#f59e0b"), unsafe_allow_html=True)
    with c4:
        st.markdown(create_kpi_card("Top-5 Control", f"{top5_c_share:.1f}%", "Monopoly Index", "🕸️", "#ec4899"), unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Ranking Matrix", "Distribution Profile", "Contribution Treemap"])
    
    with tab1:
        fig = px.bar(
            comm_df.head(top_n)[::-1],
            x=target_comm,
            y="Provinsi",
            orientation='h',
            text=target_comm,
            title=f"Hierarchical Ranking: {target_comm}"
        )
        fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        fig = apply_dark_layout(fig, 600)
        fig.update_layout(xaxis_title="K Ton")
        st.plotly_chart(fig, use_container_width=True, key=f"comm_rank_{target_comm}_001")
        
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            fig_h = px.histogram(
                comm_df,
                x=target_comm,
                nbins=15,
                color_discrete_sequence=["#3b82f6"],
                title="Frequency Distribution"
            )
            fig_h = apply_dark_layout(fig_h, 450)
            st.plotly_chart(fig_h, use_container_width=True, key=f"comm_hist_{target_comm}_001")
        with c2:
            fig_b = px.box(
                comm_df,
                y=target_comm,
                color_discrete_sequence=["#f59e0b"],
                points="all",
                title="Outlier Detection (Boxplot)"
            )
            fig_b = apply_dark_layout(fig_b, 450)
            st.plotly_chart(fig_b, use_container_width=True, key=f"comm_box_{target_comm}_001")
            
    with tab3:
        fig_t = px.treemap(
            comm_df.head(15),
            path=["Provinsi"],
            values=target_comm,
            color=target_comm,
            color_continuous_scale="viridis",
            title="Spatial Contribution Treemap"
        )
        fig_t = apply_dark_layout(fig_t, 600)
        st.plotly_chart(fig_t, use_container_width=True, key=f"comm_tree_{target_comm}_001")
        
    st.markdown(generate_insight_box(
        f"<b>Commodity Deep-Dive:</b> <b>{target_comm}</b> shows high variance (IQR indicates regional disparity). "
        f"<b>{top_prov_c}</b> acts as the absolute anchor for this commodity."
    ), unsafe_allow_html=True)


# =========================================================
# PAGE 3: PROVINCE INTELLIGENCE
# =========================================================
elif menu == "🗺️ Province Intelligence":
    target_prov = st.selectbox("Select Province to Profile", df["Provinsi"].tolist(), key="prov_int_select")
    p_df = df[df["Provinsi"] == target_prov].iloc[0]
    
    p_profile = pd.DataFrame({
        "Commodity": numeric_cols,
        "Value": [p_df[c] for c in numeric_cols]
    }).sort_values("Value", ascending=False)
    
    tot_p = p_profile["Value"].sum()
    dom_p = p_profile.iloc[0]["Commodity"] if not p_profile.empty else "-"
    active_c = (p_profile["Value"] > 0).sum()
    
    # National Average comparison
    nat_avg = pd.DataFrame({
        "Commodity": numeric_cols,
        "Value": [df[c].mean() for c in numeric_cols]
    })
    comp_df = p_profile.merge(nat_avg, on="Commodity", suffixes=("_Prov", "_Nat"))
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(create_kpi_card("Total Output", format_num(tot_p), "Ribu Ton", "📍", "#3b82f6"), unsafe_allow_html=True)
    with c2:
        st.markdown(create_kpi_card("Core Commodity", dom_p, "Dominant", "🌾", "#10b981"), unsafe_allow_html=True)
    with c3:
        st.markdown(create_kpi_card("Active Sectors", str(active_c), f"out of {len(numeric_cols)}", "⚙️", "#f59e0b"), unsafe_allow_html=True)
    with c4:
        dom_share = (p_profile.iloc[0]['Value'] / max(1, tot_p) * 100) if tot_p > 0 else 0
        st.markdown(create_kpi_card("Specialization", f"{dom_share:.0f}%", "Dominance Share", "🎯", "#06b6d4"), unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        # Radar Chart (Polar)
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(
            r=p_profile["Value"].tolist() + [p_profile["Value"].iloc[0]],
            theta=p_profile["Commodity"].tolist() + [p_profile["Commodity"].iloc[0]],
            fill='toself',
            name=target_prov,
            line_color="#3b82f6",
            opacity=0.6
        ))
        fig_r.add_trace(go.Scatterpolar(
            r=nat_avg["Value"].tolist() + [nat_avg["Value"].iloc[0]],
            theta=nat_avg["Commodity"].tolist() + [nat_avg["Commodity"].iloc[0]],
            fill='toself',
            name="National Avg",
            line_color="#f59e0b",
            opacity=0.4
        ))
        fig_r.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, showticklabels=False, gridcolor="rgba(148,163,184,0.2)"),
                angularaxis=dict(gridcolor="rgba(148,163,184,0.2)")
            ),
            showlegend=True,
            title="Multi-Dimensional Profile (Radar)"
        )
        fig_r = apply_dark_layout(fig_r, 550)
        st.plotly_chart(fig_r, use_container_width=True, key=f"prov_radar_{target_prov}_001")
        
    with col2:
        fig_bar = px.bar(
            p_profile,
            x="Commodity",
            y="Value",
            text="Value",
            color="Value",
            color_continuous_scale="tealgrn",
            title="Absolute Production Yield"
        )
        fig_bar.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        fig_bar = apply_dark_layout(fig_bar, 550)
        st.plotly_chart(fig_bar, use_container_width=True, key=f"prov_bar_{target_prov}_001")

    # Benchmark Diverging
    st.markdown("### ⚖️ Benchmark vs National Average")
    fig_div = go.Figure()
    colors = np.where(comp_df["Value_Prov"] >= comp_df["Value_Nat"], "#10b981", "#ef4444")
    fig_div.add_trace(go.Bar(
        y=comp_df["Commodity"],
        x=comp_df["Value_Prov"] - comp_df["Value_Nat"],
        orientation='h',
        marker_color=colors,
        name="Deviation"
    ))
    fig_div.update_layout(
        title="Deviation from National Mean (Green = Above Avg, Red = Below)",
        xaxis_title="Delta (K Ton)"
    )
    fig_div = apply_dark_layout(fig_div, 450)
    st.plotly_chart(fig_div, use_container_width=True, key=f"prov_div_{target_prov}_001")


# =========================================================
# PAGE 4: COMPARATIVE ANALYTICS
# =========================================================
elif menu == "📊 Comparative Analytics":
    tab1, tab2, tab3, tab4 = st.tabs([
        "Province vs Province",
        "Commodity Relationship",
        "Correlation Matrix",
        "National Benchmark"
    ])
    
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            p1 = st.selectbox("Province A", df["Provinsi"].tolist(), index=0, key="comp_prov_a")
        with c2:
            p2 = st.selectbox("Province B", df["Provinsi"].tolist(), index=3, key="comp_prov_b")
        
        p1_data = df[df["Provinsi"] == p1].iloc[0]
        p2_data = df[df["Provinsi"] == p2].iloc[0]
        
        compare_df = pd.DataFrame({
            "Commodity": numeric_cols,
            p1: [p1_data[c] for c in numeric_cols],
            p2: [p2_data[c] for c in numeric_cols]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name=p1, x=compare_df["Commodity"], y=compare_df[p1], marker_color="#3b82f6"))
        fig.add_trace(go.Bar(name=p2, x=compare_df["Commodity"], y=compare_df[p2], marker_color="#f59e0b"))
        fig.update_layout(barmode='group', title=f"Head-to-Head: {p1} vs {p2}")
        fig = apply_dark_layout(fig, 500)
        st.plotly_chart(fig, use_container_width=True, key="comp_prov_vs_prov_001")
        
        # Summary
        p1_wins = (compare_df[p1] > compare_df[p2]).sum()
        p2_wins = (compare_df[p2] > compare_df[p1]).sum()
        st.markdown(generate_insight_box(
            f"<b>Battle Result:</b> {p1} leads in <b>{p1_wins}</b> commodities, while {p2} leads in <b>{p2_wins}</b> commodities."
        ), unsafe_allow_html=True)
    
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            x_var = st.selectbox("Variable X", numeric_cols, index=0, key="comp_x_var")
        with c2:
            y_var = st.selectbox("Variable Y", numeric_cols, index=1, key="comp_y_var")
        
        if x_var != y_var:
            fig = px.scatter(
                active_df,
                x=x_var,
                y=y_var,
                hover_name="Provinsi",
                title=f"Relationship: {x_var} vs {y_var}"
            )
            fig = apply_dark_layout(fig, 550)
            st.plotly_chart(fig, use_container_width=True, key="comp_scatter_001")
            
            corr_val = active_df[[x_var, y_var]].corr().iloc[0, 1]
            if pd.notna(corr_val):
                st.markdown(generate_insight_box(
                    f"<b>Pearson Correlation:</b> {corr_val:.3f} - "
                    f"{'Strong positive' if corr_val > 0.7 else 'Moderate' if corr_val > 0.3 else 'Weak'} relationship."
                ), unsafe_allow_html=True)
        else:
            st.warning("Please select two different variables.")
    
    with tab3:
        st.markdown("### 🔥 Correlation Heatmap")
        corr_matrix = active_df[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            zmin=-1,
            zmax=1,
            color_continuous_scale=[
                [0.0, "#dc2626"],
                [0.5, "#1e293b"],
                [1.0, "#059669"]
            ],
            title="Commodity Correlation Matrix"
        )
        fig.update_traces(textfont=dict(color="white", size=11), xgap=2, ygap=2)
        fig.update_xaxes(side="bottom", tickangle=30)
        fig.update_yaxes(autorange="reversed")
        fig = apply_dark_layout(fig, 600)
        st.plotly_chart(fig, use_container_width=True, key="comp_heatmap_001")
        
        st.markdown(generate_insight_box(
            "Most commodities show weak correlation, indicating <b>regional specialization</b>. "
            "Each province tends to focus on specific commodities based on comparative advantages."
        ), unsafe_allow_html=True)
    
    with tab4:
        bench_prov = st.selectbox("Select Province for Benchmarking", df["Provinsi"].tolist(), key="bench_prov")
        bench_data = df[df["Provinsi"] == bench_prov].iloc[0]
        
        bench_df = pd.DataFrame({
            "Commodity": numeric_cols,
            "Province": [bench_data[c] for c in numeric_cols],
            "National_Avg": [df[c].mean() for c in numeric_cols]
        })
        bench_df["Ratio"] = bench_df["Province"] / bench_df["National_Avg"]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=bench_df["Commodity"],
            y=bench_df["Ratio"],
            marker_color=np.where(bench_df["Ratio"] >= 1, "#10b981", "#ef4444"),
            name="Performance Ratio"
        ))
        fig.add_hline(y=1, line_dash="dash", line_color="#f59e0b")
        fig.update_layout(
            title=f"{bench_prov} vs National Average (1.0 = Average)",
            yaxis_title="Performance Ratio"
        )
        fig = apply_dark_layout(fig, 500)
        st.plotly_chart(fig, use_container_width=True, key="comp_benchmark_001")


# =========================================================
# PAGE 5: PREDICTIVE ANALYTICS
# =========================================================
elif menu == "🤖 Predictive Analytics":
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Linear Regression",
        "Forecasting 2025",
        "Random Forest",
        "Decision Tree",
        "K-Means Clustering"
    ])
    
    with tab1:
        st.markdown("### 📉 Linear Regression Playground")
        c1, c2 = st.columns(2)
        with c1:
            x_var = st.selectbox("Independent Variable (X)", numeric_cols, index=1, key="lr_x")
        with c2:
            y_var = st.selectbox("Dependent Variable (Y)", numeric_cols, index=0, key="lr_y")
        
        if x_var != y_var:
            model_df = active_df[[x_var, y_var]].dropna()
            if len(model_df) >= 3:
                X = model_df[[x_var]].values
                y = model_df[y_var].values
                
                lr = LinearRegression()
                lr.fit(X, y)
                y_pred = lr.predict(X)
                
                mae = mean_absolute_error(y, y_pred)
                rmse = math.sqrt(mean_squared_error(y, y_pred))
                r2 = r2_score(y, y_pred)
                
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("MAE", f"{mae:.2f}")
                with c2: st.metric("RMSE", f"{rmse:.2f}")
                with c3: st.metric("R²", f"{r2:.4f}")
                
                st.markdown(generate_insight_box(
                    f"<b>Model Equation:</b> {y_var} = {lr.coef_[0]:.4f} × {x_var} + {lr.intercept_:.4f}"
                ), unsafe_allow_html=True)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=model_df[x_var], y=model_df[y_var], mode="markers", name="Actual", marker=dict(color="#3b82f6", size=10)))
                sort_idx = np.argsort(model_df[x_var].values)
                fig.add_trace(go.Scatter(x=model_df[x_var].values[sort_idx], y=y_pred[sort_idx], mode="lines", name="Regression Line", line=dict(color="#10b981", width=3)))
                fig.update_layout(title=f"Linear Regression: {x_var} vs {y_var}")
                fig = apply_dark_layout(fig, 500)
                st.plotly_chart(fig, use_container_width=True, key="lr_chart_001")
            else:
                st.warning("Insufficient data for regression.")
        else:
            st.warning("X and Y must be different variables.")
    
    with tab2:
        st.markdown("### 📈 Forecasting 2025 (Growth Rate Simulation)")
        st.markdown(generate_insight_box(
            "This dataset is cross-sectional (1 year). Forecasting uses <b>growth rate simulation</b>, not time-series."
        , "warning"), unsafe_allow_html=True)
        
        fc_comm = st.selectbox("Select Commodity", numeric_cols, key="fc_comm")
        growth = st.slider("Growth Rate (%)", 1, 20, 7, key="fc_growth") / 100
        
        fc_df = active_df[["Provinsi", fc_comm]].copy()
        fc_df["Forecast_2025"] = fc_df[fc_comm] * (1 + growth)
        fc_df["Increase"] = fc_df["Forecast_2025"] - fc_df[fc_comm]
        
        top_fc = fc_df.sort_values(fc_comm, ascending=False).head(top_n)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=top_fc["Provinsi"], y=top_fc[fc_comm], name="2024", marker_color="#3b82f6"))
        fig.add_trace(go.Bar(x=top_fc["Provinsi"], y=top_fc["Forecast_2025"], name="2025 (Forecast)", marker_color="#10b981"))
        fig.update_layout(barmode="group", title=f"{fc_comm}: 2024 vs 2025 Forecast")
        fig = apply_dark_layout(fig, 500)
        st.plotly_chart(fig, use_container_width=True, key="fc_chart_001")
        
        total_2024 = fc_df[fc_comm].sum()
        total_2025 = fc_df["Forecast_2025"].sum()
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Total 2024", format_num(total_2024))
        with c2: st.metric("Total 2025", format_num(total_2025))
        with c3: st.metric("Increase", format_num(total_2025 - total_2024))
    
    with tab3:
        st.markdown("### 🌲 Random Forest Regression")
        target_rf = st.selectbox("Target Variable", numeric_cols, key="rf_target")
        features = [c for c in numeric_cols if c != target_rf]
        
        rf_df = active_df[features + [target_rf]].dropna()
        if len(rf_df) >= 8:
            X = rf_df[features]
            y = rf_df[target_rf]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = math.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("MAE", f"{mae:.2f}")
            with c2: st.metric("RMSE", f"{rmse:.2f}")
            with c3: st.metric("R²", f"{r2:.4f}")
            
            importance = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True)
            fig = px.bar(x=importance.values, y=importance.index, orientation='h', title=f"Feature Importance - {target_rf}")
            fig = apply_dark_layout(fig, 500)
            st.plotly_chart(fig, use_container_width=True, key="rf_chart_001")
        else:
            st.warning("Insufficient data for Random Forest.")
    
    with tab4:
        st.markdown("### 🌳 Decision Tree Regression")
        target_dt = st.selectbox("Target Variable", numeric_cols, key="dt_target")
        features_dt = [c for c in numeric_cols if c != target_dt]
        
        dt_df = active_df[features_dt + [target_dt]].dropna()
        if len(dt_df) >= 8:
            X = dt_df[features_dt]
            y = dt_df[target_dt]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            dt = DecisionTreeRegressor(max_depth=3, random_state=42)
            dt.fit(X_train, y_train)
            y_pred = dt.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = math.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("MAE", f"{mae:.2f}")
            with c2: st.metric("RMSE", f"{rmse:.2f}")
            with c3: st.metric("R²", f"{r2:.4f}")
            
            importance = pd.Series(dt.feature_importances_, index=features_dt).sort_values(ascending=True)
            fig = px.bar(x=importance.values, y=importance.index, orientation='h', title=f"Feature Importance - {target_dt}")
            fig = apply_dark_layout(fig, 500)
            st.plotly_chart(fig, use_container_width=True, key="dt_chart_001")
        else:
            st.warning("Insufficient data for Decision Tree.")
    
    with tab5:
        st.markdown("### 🎯 K-Means Clustering")
        st.markdown(generate_insight_box(
            "Clustering provinces based on production patterns to identify regional segments."
        ), unsafe_allow_html=True)
        
        n_clusters = st.slider("Number of Clusters", 2, 6, 4, key="km_clusters")
        
        cluster_df = df[numeric_cols].copy()
        scaler = StandardScaler()
        cluster_scaled = scaler.fit_transform(cluster_df)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(cluster_scaled)
        
        df_cluster = df.copy()
        df_cluster["Cluster"] = clusters
        
        # PCA for 2D visualization
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(cluster_scaled)
        df_cluster["PCA1"] = pca_result[:, 0]
        df_cluster["PCA2"] = pca_result[:, 1]
        
        fig = px.scatter(
            df_cluster,
            x="PCA1",
            y="PCA2",
            color="Cluster",
            hover_name="Provinsi",
            title="Province Clusters (PCA 2D Projection)"
        )
        fig = apply_dark_layout(fig, 550)
        st.plotly_chart(fig, use_container_width=True, key="km_chart_001")
        
        # Cluster summary
        cluster_summary = df_cluster.groupby("Cluster")[numeric_cols].mean()
        st.dataframe(cluster_summary, use_container_width=True)


# =========================================================
# PAGE 6: DATA & EXPORT
# =========================================================
elif menu == "📦 Data & Export":
    st.markdown("### 📊 Dataset Overview")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total Observations", len(active_df))
    with c2: st.metric("Total Variables", len(active_df.columns))
    with c3: st.metric("Missing Values", int(active_df.isnull().sum().sum()))
    with c4: st.metric("Duplicates", int(active_df.duplicated().sum()))
    
    tab1, tab2, tab3 = st.tabs(["Dataset", "Descriptive Statistics", "Export"])
    
    with tab1:
        st.dataframe(active_df, use_container_width=True, hide_index=True)
    
    with tab2:
        desc = active_df[numeric_cols].describe().T
        desc["range"] = desc["max"] - desc["min"]
        st.dataframe(desc, use_container_width=True)
    
    with tab3:
        st.markdown("### 📥 Download Options")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Filtered Dataset")
            export_data = add_total_production(active_df)
            st.download_button(
                label="⬇️ Download Filtered Data (CSV)",
                data=export_csv(export_data),
                file_name="filtered_dataset.csv",
                mime="text/csv",
                use_container_width=True,
                key="export_filtered_btn"
            )
        
        with col2:
            st.markdown("#### Descriptive Statistics")
            st.download_button(
                label="⬇️ Download Statistics (CSV)",
                data=export_csv(desc),
                file_name="descriptive_statistics.csv",
                mime="text/csv",
                use_container_width=True,
                key="export_stats_btn"
            )
        
        st.markdown("---")
        st.markdown("### 📝 Data Quality Notes")
        st.markdown(generate_insight_box(
            "<b>✅ Data Quality:</b> Dataset is clean with no missing values or duplicates. "
            "Outliers in Kelapa Sawit (Riau, Kalteng) represent actual production centers, not errors."
        , "success"), unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #64748b; font-size: 0.9rem;">
    <h3 style="color: #f8fafc; margin-bottom: 0.5rem;">🎓 Indonesian Plantation Intelligence Dashboard</h3>
    <p>UAS Pengenalan Sains Data — Visualisasi Data & Analisis Data Dasar</p>
    <p style="margin-top: 1rem;">Built with Streamlit, Plotly, and Scikit-Learn</p>
    <p>© 2026 | Data Source: BPS - Produksi Tanaman Perkebunan 2024</p>
</div>
""", unsafe_allow_html=True)
