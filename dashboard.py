import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# STYLE
plt.style.use('default')
sns.set_theme(style="whitegrid")

st.set_page_config(page_title="Student Dropout Dashboard", layout="wide")

st.title("🎓 Business Dashboard - Student Dropout Analysis")
st.markdown("### 📊 Monitoring Faktor Penyebab Dropout Mahasiswa")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv", sep=';')
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    return df

df = load_data()

# =========================
# VALIDASI KOLOM
# =========================
required_cols = [
    'Status',
    'Tuition_fees_up_to_date',
    'Scholarship_holder',
    'Curricular_units_1st_sem_grade',
    'Curricular_units_2nd_sem_grade'
]

for col in required_cols:
    if col not in df.columns:
        st.error(f"Kolom '{col}' tidak ditemukan! Kolom tersedia: {df.columns}")
        st.stop()

# =========================
# FILTER DASHBOARD (SEMUA STATUS BOLEH)
# =========================
st.sidebar.header("Filter Data")

status_filter = st.sidebar.multiselect(
    "Pilih Status",
    options=df['Status'].unique(),
    default=df['Status'].unique()
)

df_filtered = df[df['Status'].isin(status_filter)]

# =========================
# KPI
# =========================
total = len(df_filtered)
dropout = len(df_filtered[df_filtered['Status'] == 'Dropout'])
graduate = len(df_filtered[df_filtered['Status'] == 'Graduate'])

dropout_rate = (dropout / total) * 100 if total > 0 else 0

col1, col2, col3 = st.columns(3)

col1.metric("Total Mahasiswa", total)
col2.metric("Dropout", dropout)
col3.metric("Dropout Rate", f"{dropout_rate:.2f}%")

st.markdown("---")

# =========================
# VISUALISASI
# =========================
st.subheader("📊 Visualisasi Data")

df_vis = df_filtered.copy()

# Ubah biner ke Yes/No
binary_cols = ['Tuition_fees_up_to_date', 'Scholarship_holder']
for col in binary_cols:
    df_vis[col] = df_vis[col].map({0: "No", 1: "Yes"})

# =========================
# ROW 1
# =========================
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 1. Distribusi Status")
    fig1, ax1 = plt.subplots()
    df_vis['Status'].value_counts().plot(kind='bar', ax=ax1)
    st.pyplot(fig1)

with col2:
    st.markdown("### 2. Pengaruh Pembayaran")
    fig2, ax2 = plt.subplots()
    sns.countplot(data=df_vis, x='Tuition_fees_up_to_date', hue='Status', ax=ax2)
    st.pyplot(fig2)

# =========================
# ROW 2
# =========================
col3, col4 = st.columns(2)

with col3:
    st.markdown("### 3. Pengaruh Beasiswa")
    fig3, ax3 = plt.subplots()
    sns.countplot(data=df_vis, x='Scholarship_holder', hue='Status', ax=ax3)
    st.pyplot(fig3)

with col4:
    st.markdown("### 4. Nilai Semester 1")
    fig4, ax4 = plt.subplots()
    sns.boxplot(data=df_vis, x='Status', y='Curricular_units_1st_sem_grade', ax=ax4)
    st.pyplot(fig4)

# =========================
# ROW 3
# =========================
col5, col6 = st.columns(2)

with col5:
    st.markdown("### 5. Nilai Semester 2")
    fig5, ax5 = plt.subplots()
    sns.boxplot(data=df_vis, x='Status', y='Curricular_units_2nd_sem_grade', ax=ax5)
    st.pyplot(fig5)

# =========================
# FEATURE IMPORTANCE (SUDAH FIX)
# =========================
with col6:
    st.markdown("### 6. Faktor Utama Dropout")

    try:
        df_model = df.copy()

        # 🔥 FILTER WAJIB
        df_model = df_model[df_model['Status'].isin(['Dropout', 'Graduate'])]

        # 🔥 LABEL BINER
        df_model['Status'] = df_model['Status'].map({
            'Dropout': 1,
            'Graduate': 0
        })

        X = df_model.select_dtypes(include=['int64', 'float64']).drop(columns=['Status'], errors='ignore')
        y = df_model['Status']

        X = X.fillna(0)

        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)

        feat_importance = pd.Series(model.feature_importances_, index=X.columns)
        feat_importance = feat_importance.sort_values(ascending=False)

        fig6, ax6 = plt.subplots()
        feat_importance.head(10).plot(kind='barh', ax=ax6)

        ax6.set_title("Top 10 Faktor Dropout")
        ax6.invert_yaxis()

        st.pyplot(fig6)

    except Exception as e:
        st.error(f"Gagal membuat feature importance: {e}")

st.markdown("---")
