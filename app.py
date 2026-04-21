import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Sayfa Yapılandırması
st.set_page_config(page_title="Personality Predictor", page_icon="🧠")

# Modeli Yükle
@st.cache_resource
def load_model():
    return joblib.load('adaboost_personality_model.pkl')

model = load_model()

st.title("🧠 Kişilik Tipi Tahmin Uygulaması")
st.write("Aşağıdaki bilgileri doldurarak içedönük mü yoksa dışadönük mü olduğunuzu öğrenin.")

# Kullanıcı Girişleri
col1, col2 = st.columns(2)

with col1:
    alone_time = st.slider("Yalnız Geçirilen Zaman (Saat)", 0, 10, 5)
    social_events = st.slider("Sosyal Etkinlik Katılımı", 0, 10, 3)
    going_outside = st.slider("Dışarı Çıkma Sıklığı", 0, 10, 4)

with col2:
    friends_size = st.number_input("Arkadaş Çevresi Boyutu", 0, 100, 10)
    post_freq = st.number_input("Sosyal Medya Paylaşım Sıklığı", 0, 50, 5)
    stage_fear = st.selectbox("Sahne Korkusu Var mı?", ["Hayır", "Evet"])
    drained = st.selectbox("Sosyalleşince Enerjiniz Tükenir mi?", ["Hayır", "Evet"])

# Veriyi Hazırla (Modelin beklediği formatta)
input_data = pd.DataFrame({
    'Time_spent_Alone': [alone_time],
    'Social_event_attendance': [social_events],
    'Going_outside': [going_outside],
    'Friends_circle_size': [friends_size],
    'Post_frequency': [post_freq],
    'Stage_fear_Yes': [1 if stage_fear == "Evet" else 0],
    'Drained_after_socializing_Yes': [1 if drained == "Evet" else 0]
})

# Tahmin Butonu
if st.button("Tahmin Et"):
    prediction = model.predict(input_data)
    result = "Extrovert (Dışa Dönük)" if prediction[0] == 1 else "Introvert (İçe Dönük)"
    
    st.success(f"Tahmin Edilen Kişilik Tipi: **{result}**")
    
    # Olasılıkları Göster
    probs = model.predict_proba(input_data)
    st.info(f"Olasılık: %{np.max(probs)*100:.2f}")