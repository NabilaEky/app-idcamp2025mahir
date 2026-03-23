import streamlit as st
import pandas as pd
import pickle

# LOAD MODEL & FEATURE
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/features.pkl", "rb") as f:
    features = pickle.load(f)

# CONFIG PAGE
st.set_page_config(page_title="Prediksi Dropout", layout="centered")

# TITLE
st.title("🎓 Prediksi Risiko Dropout Siswa")
st.markdown("""
Sistem ini membantu mendeteksi siswa yang berpotensi **dropout lebih awal**  
berdasarkan **performa akademik** dan **kondisi finansial**.
""")

st.divider()

# FORM INPUT
with st.form("form_prediksi"):

    st.subheader("📊 Data Akademik")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Usia Saat Masuk", 15, 60, 20)
        admission_grade = st.number_input("Nilai Masuk", 0.0, 200.0, 120.0)
        sem1_grade = st.number_input("Nilai Semester 1", 0.0, 20.0, 10.0)

    with col2:
        sem2_grade = st.number_input("Nilai Semester 2", 0.0, 20.0, 10.0)
        sem1_approved = st.number_input("Mata Kuliah Lulus Semester 1", 0, 20, 5)
        sem2_approved = st.number_input("Mata Kuliah Lulus Semester 2", 0, 20, 5)

    st.subheader("💰 Kondisi Finansial")

    col3, col4 = st.columns(2)

    with col3:
        tuition_label = st.selectbox("Status Pembayaran", ["Lunas", "Belum Lunas"])

    with col4:
        scholarship_label = st.selectbox("Status Beasiswa", ["Penerima Beasiswa", "Tidak Menerima"])

    submit = st.form_submit_button("Prediksi")

# MAPPING INPUT
tuition = 1 if tuition_label == "Lunas" else 0
scholarship = 1 if scholarship_label == "Penerima Beasiswa" else 0

input_data = pd.DataFrame([{
    'Age_at_enrollment': age,
    'Admission_grade': admission_grade,
    'Curricular_units_1st_sem_grade': sem1_grade,
    'Curricular_units_2nd_sem_grade': sem2_grade,
    'Curricular_units_1st_sem_approved': sem1_approved,
    'Curricular_units_2nd_sem_approved': sem2_approved,
    'Tuition_fees_up_to_date': tuition,
    'Scholarship_holder': scholarship
}])

# Sesuaikan fitur model
for col in features:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[features]

# HASIL PREDIKSI
if submit:
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    st.divider()
    st.subheader("📢 Hasil Prediksi")

    # LEVEL RISIKO
    if proba > 0.7:
        risk_level = "Tinggi"
    elif proba > 0.4:
        risk_level = "Sedang"
    else:
        risk_level = "Rendah"

    colA, colB = st.columns(2)
    colA.metric("Probabilitas Dropout", f"{proba*100:.2f}%")
    colB.metric("Level Risiko", risk_level)

    # STATUS
    if prediction == 1:
        st.error("Siswa berisiko dropout")
    else:
        st.success("Siswa tidak berisiko dropout")

    # REKOMENDASI CERDAS
    rekomendasi = []

    # 🔴 Risiko Tinggi → detail
    if risk_level == "Tinggi":

        if sem1_grade < 8:
            rekomendasi.append("Nilai Semester 1 rendah, perlu bimbingan akademik intensif.")
        elif sem1_grade <= 12:
            rekomendasi.append("Nilai Semester 1 cukup, perlu peningkatan.")

        if sem2_grade < 8:
            rekomendasi.append("Nilai Semester 2 rendah, perlu intervensi segera.")
        elif sem2_grade <= 12:
            rekomendasi.append("Nilai Semester 2 cukup, perlu peningkatan.")

        if sem1_approved <= 6:
            rekomendasi.append("Jumlah mata kuliah lulus semester 1 perlu ditingkatkan.")

        if sem2_approved <= 6:
            rekomendasi.append("Jumlah mata kuliah lulus semester 2 perlu ditingkatkan.")

        if tuition == 0:
            rekomendasi.append("Status pembayaran belum lunas, perlu perhatian.")

        if scholarship == 0:
            rekomendasi.append("Disarankan mempertimbangkan bantuan beasiswa.")

    # 🟡 Risiko Sedang → fokus utama
    elif risk_level == "Sedang":

        if sem2_grade <= 10:
            rekomendasi.append("Performa semester 2 perlu ditingkatkan.")

        if sem2_approved <= 5:
            rekomendasi.append("Jumlah mata kuliah lulus semester 2 perlu ditingkatkan.")

        if tuition == 0:
            rekomendasi.append("Perlu perhatian pada kondisi pembayaran.")

    # 🟢 Risiko Rendah → minimal
    else:
        rekomendasi.append("Performa siswa cukup baik, tetap lakukan monitoring secara berkala.")

    # TAMPILKAN
    st.markdown("### Rekomendasi Tindakan")
    st.write("\n".join([f"- {r}" for r in rekomendasi]))