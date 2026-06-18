import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Prediksi Harga Beras Indonesia",
    page_icon="🌾",
    layout="wide"
)

# ============================================================
# LOAD MODEL & DATA (cached supaya tidak reload setiap interaksi)
# ============================================================
@st.cache_resource
def load_model():
    with open('model_rf_beras.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoder_province.pkl', 'rb') as f:
        le_province = pickle.load(f)
    with open('provinsi_list.pkl', 'rb') as f:
        provinsi_list = pickle.load(f)
    return model, le_province, provinsi_list

@st.cache_data
def load_historical_data():
    df = pd.read_csv('data_historis_beras.csv')
    df['Date_Param'] = pd.to_datetime(df['Date_Param'])
    return df

model, le_province, provinsi_list = load_model()
df_hist = load_historical_data()

# ============================================================
# HEADER
# ============================================================
st.title("🌾 Prediksi Harga Beras di Indonesia")
st.markdown(
    "Aplikasi ini memprediksi **harga beras (Rp/kg)** berdasarkan **tanggal** dan **provinsi** "
    "menggunakan model **Random Forest Regression** (R² = 99.93% pada data uji)."
)
st.divider()

# ============================================================
# SIDEBAR INPUT
# ============================================================
st.sidebar.header("⚙️ Parameter Prediksi")

tanggal_input = st.sidebar.date_input(
    "📅 Pilih Tanggal",
    value=datetime.date(2026, 6, 18),
    min_value=datetime.date(2022, 1, 1),
    max_value=datetime.date(2027, 12, 31)
)

provinsi_input = st.sidebar.selectbox(
    "🗺️ Pilih Provinsi",
    options=provinsi_list,
    index=provinsi_list.index("Jawa Timur") if "Jawa Timur" in provinsi_list else 0
)

predict_button = st.sidebar.button("🔮 Prediksi Harga", type="primary", use_container_width=True)

st.sidebar.divider()
st.sidebar.info(
    "**Tentang Model**\n\n"
    "- Algoritma: Random Forest Regression\n"
    "- Data: PIHPS Bank Indonesia (2022–2026)\n"
    "- Cakupan: 34 Provinsi\n"
    "- Fitur: Tahun, Bulan, Kuartal, Hari ke-N, Provinsi"
)

# ============================================================
# FUNGSI PREDIKSI
# ============================================================
def predict_price(tanggal, provinsi):
    year = tanggal.year
    month = tanggal.month
    quarter = (month - 1) // 3 + 1
    day_of_year = tanggal.timetuple().tm_yday
    province_enc = le_province.transform([provinsi])[0]

    X_new = pd.DataFrame([{
        'Year': year,
        'Month': month,
        'Quarter': quarter,
        'DayOfYear': day_of_year,
        'Province_Enc': province_enc
    }])

    pred = model.predict(X_new)[0]
    return pred

# ============================================================
# AREA HASIL PREDIKSI
# ============================================================
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Hasil Prediksi")

    if predict_button or True:  # selalu tampilkan prediksi default
        harga_pred = predict_price(tanggal_input, provinsi_input)

        st.metric(
            label=f"Prediksi Harga Beras — {provinsi_input}",
            value=f"Rp {harga_pred:,.0f} / kg"
        )
        st.caption(f"📅 Tanggal: {tanggal_input.strftime('%d %B %Y')}")

        # Bandingkan dengan rata-rata historis provinsi tersebut
        avg_provinsi = df_hist[df_hist['Province_Name'] == provinsi_input]['Price'].mean()
        selisih = harga_pred - avg_provinsi
        st.caption(
            f"📈 Rata-rata historis {provinsi_input}: Rp {avg_provinsi:,.0f}/kg "
            f"({'lebih tinggi' if selisih > 0 else 'lebih rendah'} Rp {abs(selisih):,.0f})"
        )

with col2:
    st.subheader(f"📈 Tren Historis Harga Beras — {provinsi_input}")

    data_provinsi = df_hist[df_hist['Province_Name'] == provinsi_input].sort_values('Date_Param')
    data_provinsi_monthly = data_provinsi.groupby(
        pd.Grouper(key='Date_Param', freq='ME')
    )['Price'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(data_provinsi_monthly['Date_Param'], data_provinsi_monthly['Price'],
            color='#e74c3c', linewidth=2)
    ax.scatter([pd.Timestamp(tanggal_input)], [harga_pred],
               color='#2c3e50', s=100, zorder=5, label='Prediksi Tanggal Dipilih')
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Harga (Rp/kg)")
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

st.divider()

# ============================================================
# PERBANDINGAN ANTAR PROVINSI (opsional, tambahan insight)
# ============================================================
st.subheader("🗺️ Perbandingan Prediksi Harga Antar Provinsi (Tanggal yang Sama)")

with st.spinner("Menghitung prediksi semua provinsi..."):
    hasil_semua_provinsi = []
    for prov in provinsi_list:
        harga = predict_price(tanggal_input, prov)
        hasil_semua_provinsi.append({'Provinsi': prov, 'Prediksi Harga (Rp/kg)': harga})

    df_perbandingan = pd.DataFrame(hasil_semua_provinsi).sort_values(
        'Prediksi Harga (Rp/kg)', ascending=False
    )

fig2, ax2 = plt.subplots(figsize=(10, 9))
colors = ['#e74c3c' if p == provinsi_input else '#3498db' for p in df_perbandingan['Provinsi']]
ax2.barh(df_perbandingan['Provinsi'], df_perbandingan['Prediksi Harga (Rp/kg)'], color=colors)
ax2.set_xlabel("Prediksi Harga (Rp/kg)")
ax2.set_title(f"Prediksi Harga Beras 34 Provinsi — {tanggal_input.strftime('%d %B %Y')}")
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
plt.tight_layout()
st.pyplot(fig2)

st.caption("🔴 Provinsi yang sedang dipilih ditandai warna merah.")

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "Dibuat untuk Tugas UAS Machine Learning | "
    "Model: Random Forest Regression | "
    "Sumber Data: PIHPS Bank Indonesia"
)
