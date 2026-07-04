
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
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Dashboard Komoditas Perkebunan Indonesia",
    page_icon="🌾",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
    .main {
        background-color: #f8fafc;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        color: #1f2937;
    }

    .hero-subtitle {
        color: #475569;
        font-size: 1rem;
        margin-bottom: 1rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #1f2937;
        margin-top: 0.8rem;
        margin-bottom: 0.6rem;
    }

    .soft-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
    }

    .info-box {
        background: #eff6ff;
        border-left: 4px solid #2563eb;
        padding: 0.9rem 1rem;
        border-radius: 10px;
        margin: 0.5rem 0 1rem 0;
        color: #1e3a8a;
    }

    .warn-box {
        background: #fff7ed;
        border-left: 4px solid #f97316;
        padding: 0.9rem 1rem;
        border-radius: 10px;
        margin: 0.5rem 0 1rem 0;
        color: #9a3412;
    }

    .success-box {
        background: #ecfdf5;
        border-left: 4px solid #10b981;
        padding: 0.9rem 1rem;
        border-radius: 10px;
        margin: 0.5rem 0 1rem 0;
        color: #065f46;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5e7eb;
        padding: 0.8rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
    }

    .footer-box {
        text-align:center;
        padding:1.2rem;
        background: linear-gradient(90deg, #0f766e 0%, #2563eb 100%);
        border-radius: 14px;
        color:white;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# DATASET
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

@st.cache_data
def load_data():
    df_ = pd.read_csv(io.StringIO(CSV_DATA))
    return df_

df = load_data()
numeric_cols = [c for c in df.columns if c != "Provinsi"]

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def safe_r2(y_true, y_pred):
    try:
        return r2_score(y_true, y_pred)
    except Exception:
        return np.nan

def safe_mae(y_true, y_pred):
    try:
        return mean_absolute_error(y_true, y_pred)
    except Exception:
        return np.nan

def safe_rmse(y_true, y_pred):
    try:
        return math.sqrt(mean_squared_error(y_true, y_pred))
    except Exception:
        return np.nan

def get_filtered_df(base_df, province_filter, show_zero_rows):
    data = base_df.copy()
    if province_filter != "Semua Provinsi":
        data = data[data["Provinsi"] == province_filter].copy()

    if not show_zero_rows:
        # Hanya sembunyikan jika semua komoditas 0
        data = data[(data[numeric_cols].sum(axis=1) > 0)].copy()

    return data

def total_production_by_province(data):
    temp = data.copy()
    temp["Total Produksi"] = temp[numeric_cols].sum(axis=1)
    return temp

def commodity_summary(data):
    sums = data[numeric_cols].sum().sort_values(ascending=False)
    return sums

def province_with_highest_total(data):
    temp = total_production_by_province(data)
    if temp.empty:
        return "-", 0.0
    idx = temp["Total Produksi"].idxmax()
    return temp.loc[idx, "Provinsi"], temp.loc[idx, "Total Produksi"]

def top_province_for_commodity(data, commodity):
    if data.empty:
        return "-", 0.0
    idx = data[commodity].idxmax()
    return data.loc[idx, "Provinsi"], data.loc[idx, commodity]

def format_num(x):
    return f"{x:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")

def generate_dynamic_insight(data, commodity):
    if data.empty:
        return "Tidak ada data yang tersedia untuk filter saat ini."

    total = data[commodity].sum()
    if total <= 0:
        return f"Komoditas **{commodity}** tidak memiliki produksi pada filter saat ini."

    top_df = data.nlargest(3, commodity)[["Provinsi", commodity]].copy()
    top_df["share"] = (top_df[commodity] / total) * 100

    top1 = top_df.iloc[0]
    msg = (
        f"Produksi **{commodity}** pada filter saat ini mencapai **{format_num(total)} ribu ton**. "
        f"Kontributor terbesar adalah **{top1['Provinsi']}** dengan produksi **{format_num(top1[commodity])} ribu ton** "
        f"atau sekitar **{top1['share']:.1f}%** dari total produksi {commodity.lower()}."
    )

    if len(top_df) >= 3:
        share3 = top_df["share"].sum()
        msg += f" Tiga provinsi teratas menyumbang sekitar **{share3:.1f}%** dari total, menunjukkan tingkat konsentrasi produksi yang cukup tinggi."
    return msg

def export_csv(dataframe):
    return dataframe.to_csv(index=False).encode("utf-8")

# =========================================================
# HEADER
# =========================================================
st.markdown('<div class="hero-title">🌾 Dashboard Komoditas Perkebunan Indonesia</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Visualisasi, eksplorasi, dan pemodelan data produksi komoditas perkebunan Indonesia per provinsi (2024)</div>',
    unsafe_allow_html=True
)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("📌 Navigasi Dashboard")

menu = st.sidebar.radio(
    "Pilih Halaman",
    [
        "🏠 Executive Dashboard",
        "📊 Data Explorer",
        "🔍 EDA Explorer",
        "🧭 Profil Provinsi & Komoditas",
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

filtered_df = get_filtered_df(df, selected_province, show_zero_rows)

# =========================================================
# EXECUTIVE DASHBOARD
# =========================================================
if menu == "🏠 Executive Dashboard":
    st.markdown('<div class="section-title">Ringkasan Eksekutif</div>', unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("Tidak ada data yang cocok dengan filter saat ini.")
    else:
        total_prod = filtered_df[numeric_cols].sum().sum()
        total_prov = filtered_df.shape[0]
        top_commodity_name = filtered_df[numeric_cols].sum().idxmax()
        top_commodity_val = filtered_df[numeric_cols].sum().max()
        best_province, best_total = province_with_highest_total(filtered_df)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Provinsi", total_prov)
        c2.metric("Total Produksi", f"{format_num(total_prod)}")
        c3.metric("Komoditas Terbesar", top_commodity_name, f"{format_num(top_commodity_val)}")
        c4.metric("Provinsi Tertinggi", best_province, f"{format_num(best_total)}")

        st.markdown("")

        left, right = st.columns([1.15, 1])

        with left:
            st.markdown('<div class="section-title">Top Provinsi Berdasarkan Komoditas Terpilih</div>', unsafe_allow_html=True)
            top_df = filtered_df.nlargest(min(top_n, len(filtered_df)), selected_commodity)[["Provinsi", selected_commodity]].copy()

            fig = px.bar(
                top_df.sort_values(selected_commodity, ascending=True),
                x=selected_commodity,
                y="Provinsi",
                orientation="h",
                text_auto=".2s",
                title=f"Top {min(top_n, len(filtered_df))} Provinsi - {selected_commodity}"
            )
            fig.update_layout(height=520, xaxis_title="Produksi (Ribu Ton)", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.markdown('<div class="section-title">Komposisi Produksi Antar Komoditas</div>', unsafe_allow_html=True)
            commodity_sum = filtered_df[numeric_cols].sum().reset_index()
            commodity_sum.columns = ["Komoditas", "Produksi"]

            fig2 = px.pie(
                commodity_sum,
                values="Produksi",
                names="Komoditas",
                hole=0.45
            )
            fig2.update_layout(height=520)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-title">Insight Cepat</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="info-box">{generate_dynamic_insight(filtered_df, selected_commodity)}</div>', unsafe_allow_html=True)

        # tabel ringkas
        st.markdown('<div class="section-title">Tabel Ringkasan Produksi per Provinsi</div>', unsafe_allow_html=True)
        summary_table = total_production_by_province(filtered_df)[["Provinsi"] + numeric_cols + ["Total Produksi"]]
        st.dataframe(summary_table, use_container_width=True)

# =========================================================
# DATA EXPLORER
# =========================================================
elif menu == "📊 Data Explorer":
    st.markdown('<div class="section-title">Data Explorer</div>', unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("Tidak ada data untuk filter yang dipilih.")
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
            zero_total = (filtered_df[numeric_cols].sum(axis=1) == 0).sum()

            st.markdown(f"- **Missing values:** {int(filtered_df.isnull().sum().sum())}")
            st.markdown(f"- **Baris duplikat:** {int(duplicate_count)}")
            st.markdown(f"- **Provinsi dengan total produksi 0:** {int(zero_total)}")
            st.markdown(
                '<div class="warn-box">Outlier pada komoditas tertentu tidak otomatis dihapus karena dapat merepresentasikan sentra produksi riil.</div>',
                unsafe_allow_html=True
            )

# =========================================================
# EDA EXPLORER
# =========================================================
elif menu == "🔍 EDA Explorer":
    st.markdown('<div class="section-title">EDA Explorer</div>', unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("Tidak ada data untuk ditampilkan.")
    else:
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["📌 Overview", "📊 Distribution", "🔗 Relationship", "🔥 Correlation", "📍 Province Deep Dive"]
        )

        with tab1:
            col1, col2 = st.columns([1.2, 1])

            with col1:
                top_df = filtered_df.nlargest(min(top_n, len(filtered_df)), selected_commodity)[["Provinsi", selected_commodity]].copy()
                fig = px.bar(
                    top_df.sort_values(selected_commodity, ascending=True),
                    x=selected_commodity,
                    y="Provinsi",
                    orientation="h",
                    text_auto=".2s",
                    title=f"Top {min(top_n, len(filtered_df))} Provinsi - {selected_commodity}"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                temp = filtered_df.copy()
                temp["Total Produksi"] = temp[numeric_cols].sum(axis=1)
                fig2 = px.bar(
                    temp.sort_values("Total Produksi", ascending=False).head(min(top_n, len(temp))),
                    x="Provinsi",
                    y="Total Produksi",
                    title="Top Provinsi berdasarkan Total Produksi"
                )
                fig2.update_xaxes(tickangle=45)
                fig2.update_layout(height=500)
                st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            c1, c2 = st.columns(2)

            with c1:
                fig = px.histogram(
                    filtered_df,
                    x=selected_commodity,
                    nbins=15,
                    marginal="box",
                    title=f"Distribusi {selected_commodity}"
                )
                fig.update_layout(height=480)
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                melted = filtered_df.melt(id_vars="Provinsi", value_vars=numeric_cols,
                                          var_name="Komoditas", value_name="Produksi")
                fig2 = px.box(
                    melted,
                    x="Komoditas",
                    y="Produksi",
                    color="Komoditas",
                    title="Perbandingan Persebaran Antar Komoditas"
                )
                fig2.update_xaxes(tickangle=35)
                fig2.update_layout(height=480, showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            x_var = st.selectbox("Pilih variabel X", numeric_cols, index=0, key="eda_x")
            default_y_idx = 1 if len(numeric_cols) > 1 else 0
            y_var = st.selectbox("Pilih variabel Y", numeric_cols, index=default_y_idx, key="eda_y")

            if x_var == y_var:
                st.warning("Pilih dua variabel yang berbeda untuk scatter plot.")
            else:
                fig = px.scatter(
                    filtered_df,
                    x=x_var,
                    y=y_var,
                    hover_name="Provinsi",
                    trendline=None,
                    title=f"Hubungan {x_var} vs {y_var}",
                    size_max=16
                )
                fig.update_layout(height=550)
                st.plotly_chart(fig, use_container_width=True)

                corr_val = filtered_df[[x_var, y_var]].corr().iloc[0, 1]
                if pd.notna(corr_val):
                    st.markdown(
                        f'<div class="info-box">Korelasi Pearson antara <b>{x_var}</b> dan <b>{y_var}</b> = <b>{corr_val:.3f}</b>.</div>',
                        unsafe_allow_html=True
                    )

        with tab4:
            corr = filtered_df[numeric_cols].corr()
            fig = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title="Heatmap Korelasi Antar Komoditas"
            )
            fig.update_layout(height=650)
            st.plotly_chart(fig, use_container_width=True)

        with tab5:
            province_pick = st.selectbox(
                "Pilih provinsi untuk ditinjau",
                filtered_df["Provinsi"].tolist(),
                key="province_deep_dive"
            )
            p_df = filtered_df[filtered_df["Provinsi"] == province_pick]

            if p_df.empty:
                st.warning("Provinsi tidak ditemukan pada filter saat ini.")
            else:
                row = p_df.iloc[0]
                profile = pd.DataFrame({
                    "Komoditas": numeric_cols,
                    "Produksi": [row[c] for c in numeric_cols]
                }).sort_values("Produksi", ascending=False)

                c1, c2 = st.columns([1.1, 1])
                with c1:
                    fig = px.bar(
                        profile,
                        x="Komoditas",
                        y="Produksi",
                        text_auto=".2s",
                        title=f"Profil Produksi {province_pick}"
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)

                with c2:
                    top_comm = profile.iloc[0]["Komoditas"]
                    top_val = profile.iloc[0]["Produksi"]
                    total_val = profile["Produksi"].sum()

                    st.metric("Total Produksi Provinsi", format_num(total_val))
                    st.metric("Komoditas Dominan", top_comm, format_num(top_val))

                    fig2 = px.pie(profile, names="Komoditas", values="Produksi", hole=0.4, title="Komposisi Komoditas")
                    fig2.update_layout(height=420)
                    st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# PROFIL PROVINSI & KOMODITAS
# =========================================================
elif menu == "🧭 Profil Provinsi & Komoditas":
    st.markdown('<div class="section-title">Profil Provinsi & Komoditas</div>', unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("Tidak ada data untuk filter saat ini.")
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

                fig = px.bar(prov_profile, x="Komoditas", y="Produksi", text_auto=".2s", title=f"Struktur Komoditas - {province_pick}")
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(prov_profile, use_container_width=True)

        with tab_k:
            commodity_pick = st.selectbox("Pilih komoditas", numeric_cols, key="profile_commodity")
            cdf = filtered_df[["Provinsi", commodity_pick]].sort_values(commodity_pick, ascending=False).copy()
            total = cdf[commodity_pick].sum()

            top_prov, top_val = top_province_for_commodity(filtered_df, commodity_pick)

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Nasional (filter aktif)", format_num(total))
            c2.metric("Provinsi Tertinggi", top_prov)
            c3.metric("Produksi Tertinggi", format_num(top_val))

            fig = px.bar(
                cdf.head(min(top_n, len(cdf))),
                x="Provinsi",
                y=commodity_pick,
                text_auto=".2s",
                title=f"Top {min(top_n, len(cdf))} Provinsi - {commodity_pick}"
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                f'<div class="info-box">{generate_dynamic_insight(filtered_df, commodity_pick)}</div>',
                unsafe_allow_html=True
            )

# =========================================================
# PREDICTIVE ANALYTICS
# =========================================================
elif menu == "📈 Predictive Analytics":
    st.markdown('<div class="section-title">Predictive Analytics</div>', unsafe_allow_html=True)

    if filtered_df.shape[0] < 5:
        st.warning("Data terlalu sedikit untuk pemodelan yang stabil. Gunakan lebih banyak provinsi atau ubah filter.")
    else:
        tab_lr, tab_fc, tab_rf, tab_dt = st.tabs(
            ["📉 Regresi Linear", "📈 Forecasting", "🌲 Random Forest", "🌳 Decision Tree"]
        )

        # -------------------------------------------------
        # REGRESI LINEAR
        # -------------------------------------------------
        with tab_lr:
            st.markdown("### Regresi Linear Interaktif")

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
                    st.warning("Data tidak cukup untuk regresi.")
                else:
                    X = model_df[[x_var]].values
                    y = model_df[y_var].values

                    lr = LinearRegression()
                    lr.fit(X, y)
                    y_pred = lr.predict(X)

                    mae = safe_mae(y, y_pred)
                    rmse = safe_rmse(y, y_pred)
                    r2 = safe_r2(y, y_pred)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("MAE", f"{mae:.2f}")
                    c2.metric("RMSE", f"{rmse:.2f}")
                    c3.metric("R²", f"{r2:.4f}")

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
                        name="Aktual",
                        text=filtered_df.loc[model_df.index, "Provinsi"]
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
                        yaxis_title=y_var,
                        height=550
                    )
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
                '<div class="warn-box">Karena dataset bersifat cross-sectional (1 tahun), forecasting di sini berupa simulasi berbasis growth rate, bukan time-series forecasting historis.</div>',
                unsafe_allow_html=True
            )

            commodity_target = st.selectbox("Pilih komoditas untuk diproyeksikan", numeric_cols, key="fc_comm")
            growth_rate = st.slider("Growth Rate (%)", min_value=1, max_value=15, value=7, key="fc_growth") / 100

            fc_df = filtered_df[["Provinsi", commodity_target]].copy()
            fc_df["Forecast_2025"] = fc_df[commodity_target] * (1 + growth_rate)
            fc_df["Peningkatan"] = fc_df["Forecast_2025"] - fc_df[commodity_target]

            top_fc = fc_df.sort_values(commodity_target, ascending=False).head(min(top_n, len(fc_df))).copy()

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=top_fc["Provinsi"],
                y=top_fc[commodity_target],
                name="2024"
            ))
            fig.add_trace(go.Bar(
                x=top_fc["Provinsi"],
                y=top_fc["Forecast_2025"],
                name="2025 (Forecast)"
            ))
            fig.update_layout(
                barmode="group",
                title=f"Perbandingan {commodity_target}: 2024 vs 2025",
                xaxis_title="Provinsi",
                yaxis_title="Produksi (Ribu Ton)",
                height=550
            )
            st.plotly_chart(fig, use_container_width=True)

            total_now = fc_df[commodity_target].sum()
            total_fc = fc_df["Forecast_2025"].sum()

            c1, c2, c3 = st.columns(3)
            c1.metric("Total 2024", format_num(total_now))
            c2.metric("Total 2025 (Forecast)", format_num(total_fc))
            c3.metric("Kenaikan Total", format_num(total_fc - total_now))

            st.dataframe(fc_df.sort_values("Forecast_2025", ascending=False), use_container_width=True)

        # -------------------------------------------------
        # RANDOM FOREST
        # -------------------------------------------------
        with tab_rf:
            st.markdown("### Random Forest Regression")
            target_rf = st.selectbox("Pilih target prediksi Random Forest", numeric_cols, index=0, key="rf_target")

            feature_cols = [c for c in numeric_cols if c != target_rf]
            rf_df = filtered_df[feature_cols + [target_rf]].dropna().copy()

            if len(rf_df) < 8:
                st.warning("Data terlalu sedikit untuk Random Forest yang lebih stabil.")
            else:
                X = rf_df[feature_cols]
                y = rf_df[target_rf]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )

                rf = RandomForestRegressor(
                    n_estimators=150,
                    random_state=42,
                    max_depth=5
                )
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)

                mae = safe_mae(y_test, y_pred)
                rmse = safe_rmse(y_test, y_pred)
                r2 = safe_r2(y_test, y_pred)

                c1, c2, c3 = st.columns(3)
                c1.metric("MAE", f"{mae:.2f}")
                c2.metric("RMSE", f"{rmse:.2f}")
                c3.metric("R²", f"{r2:.4f}")

                importance = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False).reset_index()
                importance.columns = ["Fitur", "Importance"]

                fig = px.bar(
                    importance,
                    x="Importance",
                    y="Fitur",
                    orientation="h",
                    title=f"Feature Importance untuk Prediksi {target_rf}",
                    text_auto=".3f"
                )
                fig.update_layout(height=500)
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
            target_dt = st.selectbox("Pilih target prediksi Decision Tree", numeric_cols, index=0, key="dt_target")
            feature_cols_dt = [c for c in numeric_cols if c != target_dt]
            dt_df = filtered_df[feature_cols_dt + [target_dt]].dropna().copy()

            if len(dt_df) < 8:
                st.warning("Data terlalu sedikit untuk Decision Tree yang lebih stabil.")
            else:
                X = dt_df[feature_cols_dt]
                y = dt_df[target_dt]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )

                dt = DecisionTreeRegressor(max_depth=3, random_state=42)
                dt.fit(X_train, y_train)
                y_pred = dt.predict(X_test)

                mae = safe_mae(y_test, y_pred)
                rmse = safe_rmse(y_test, y_pred)
                r2 = safe_r2(y_test, y_pred)

                c1, c2, c3 = st.columns(3)
                c1.metric("MAE", f"{mae:.2f}")
                c2.metric("RMSE", f"{rmse:.2f}")
                c3.metric("R²", f"{r2:.4f}")

                importance = pd.Series(dt.feature_importances_, index=feature_cols_dt).sort_values(ascending=False).reset_index()
                importance.columns = ["Fitur", "Importance"]

                fig = px.bar(
                    importance,
                    x="Importance",
                    y="Fitur",
                    orientation="h",
                    title=f"Feature Importance Decision Tree - {target_dt}",
                    text_auto=".3f"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown(
                    '<div class="info-box">Visualisasi pohon keputusan penuh sengaja tidak ditampilkan agar dashboard tetap ringan dan stabil. Sebagai gantinya, ditampilkan feature importance untuk interpretasi model.</div>',
                    unsafe_allow_html=True
                )

# =========================================================
# INSIGHT & REKOMENDASI
# =========================================================
elif menu == "🧠 Insight & Rekomendasi":
    st.markdown('<div class="section-title">Insight & Rekomendasi</div>', unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("Tidak ada data untuk filter saat ini.")
    else:
        total_by_comm = filtered_df[numeric_cols].sum().sort_values(ascending=False)
        dominant_comm = total_by_comm.index[0]
        dominant_val = total_by_comm.iloc[0]

        top_prov, top_val = top_province_for_commodity(filtered_df, selected_commodity)
        total_selected = filtered_df[selected_commodity].sum()
        share_top = (top_val / total_selected * 100) if total_selected > 0 else 0

        insights = [
            f"Komoditas dengan total produksi terbesar pada filter aktif adalah **{dominant_comm}** sebesar **{format_num(dominant_val)} ribu ton**.",
            f"Untuk komoditas **{selected_commodity}**, provinsi tertinggi adalah **{top_prov}** dengan produksi **{format_num(top_val)} ribu ton**.",
            f"Kontribusi **{top_prov}** terhadap total **{selected_commodity}** mencapai sekitar **{share_top:.1f}%**.",
            "Korelasi antar komoditas perlu dibaca hati-hati karena produksi perkebunan sangat dipengaruhi faktor geografis, iklim, dan spesialisasi wilayah.",
            "Forecasting pada dashboard ini bersifat simulatif sehingga lebih tepat dipakai untuk eksplorasi skenario daripada prediksi kebijakan final."
        ]

        st.markdown("### 💡 Insight Utama")
        for i, ins in enumerate(insights, start=1):
            st.markdown(f'<div class="info-box"><b>{i}.</b> {ins}</div>', unsafe_allow_html=True)

        st.markdown("### 🚀 Rekomendasi")
        recommendations = [
            f"Fokuskan strategi hilirisasi pada komoditas dominan seperti **{dominant_comm}** di provinsi dengan kontribusi tertinggi.",
            f"Untuk komoditas **{selected_commodity}**, evaluasi ketergantungan pada provinsi utama seperti **{top_prov}** agar risiko konsentrasi pasokan dapat dikurangi.",
            "Gunakan analisis korelasi dan feature importance sebagai dasar awal pemetaan hubungan antar komoditas, bukan sebagai bukti kausalitas.",
            "Tambahkan data historis multi-tahun agar forecasting dan evaluasi tren menjadi lebih valid.",
            "Integrasikan peta geografis atau data spasial pada versi berikutnya untuk memperkuat analisis sebaran produksi."
        ]

        for i, rec in enumerate(recommendations, start=1):
            st.markdown(f'<div class="success-box"><b>{i}.</b> {rec}</div>', unsafe_allow_html=True)

# =========================================================
# EXPORT CENTER
# =========================================================
elif menu == "📥 Export Center":
    st.markdown('<div class="section-title">Export Center</div>', unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("Tidak ada data untuk diekspor.")
    else:
        st.markdown("Unduh hasil data yang sudah mengikuti filter aktif pada sidebar.")

        base_export = filtered_df.copy()
        base_export["Total Produksi"] = base_export[numeric_cols].sum(axis=1)

        st.download_button(
            label="⬇️ Download dataset hasil filter (CSV)",
            data=export_csv(base_export),
            file_name="dataset_hasil_filter.csv",
            mime="text/csv"
        )

        # export forecast sederhana
        exp_comm = st.selectbox("Komoditas untuk file forecast", numeric_cols, key="export_comm")
        exp_growth = st.slider("Growth rate forecast (%)", 1, 15, 7, key="export_growth") / 100

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
st.markdown(
    """
    <div class="footer-box">
        <h3>🎓 Dashboard UAS Pengenalan Sains Data</h3>
        <p>Versi interaktif dengan global filter, EDA eksploratif, predictive analytics, dan export center</p>
    </div>
    """,
    unsafe_allow_html=True
)
```
