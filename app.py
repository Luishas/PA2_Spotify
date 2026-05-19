import streamlit as st
import numpy as np
import joblib

# =========================
# CONFIGURACIÓN
# =========================

st.set_page_config(
    page_title="Spotify Genre & Mood Classifier",
    page_icon="🎵",
    layout="wide"
)
# =========================
# CARGA DE MODELOS
# =========================

rf_genre = joblib.load('modelos/rf_genre.pkl')
rf_mood  = joblib.load('modelos/rf_mood.pkl')
scaler   = joblib.load('modelos/scaler.pkl')
le_genre = joblib.load('modelos/le_genre.pkl')
le_mood  = joblib.load('modelos/le_mood.pkl')

# =========================
# TÍTULO
# =========================

st.title("🎧 Spotify Genre & Mood Predictor")
st.markdown(
    """
    Ingresa las características de audio de una canción y el modelo predecirá:

    - 🎵 Género musical
    - 💭 Mood / emoción
    """
)

st.divider()

# =========================
# SIDEBAR
# =========================

st.sidebar.header("⚙️ Parámetros de audio")

valence = st.sidebar.slider(
    "Valence (felicidad)",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01
)

energy = st.sidebar.slider(
    "Energy",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01
)
acousticness = st.sidebar.slider(
    max_value=1.0,
    value=0.5,
    step=0.01
)

danceability = st.sidebar.slider(
    "Danceability",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01
)

tempo = st.sidebar.slider(
    "Tempo",
    min_value=50.0,
    max_value=250.0,
    value=120.0,
    step=1.0
)

loudness = st.sidebar.slider(
    "Loudness",
    min_value=-60.0,
    max_value=5.0,
    value=-10.0,
    step=0.5
)

speechiness = st.sidebar.slider(
    "Speechiness",
    min_value=0.0,
    max_value=1.0,
    value=0.1,
    step=0.01
)

instrumentalness = st.sidebar.slider(
    "Instrumentalness",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.01
)

liveness = st.sidebar.slider(
    "Liveness",
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.01
)

duration_ms = st.sidebar.slider(
    "Duración (ms)",
    min_value=30000,
    max_value=600000,
    value=180000,
    step=1000
)

explicit = st.sidebar.selectbox(
    "Explicit",
    options=[0, 1],
    format_func=lambda x: "Sí" if x == 1 else "No"
)

time_signature = st.sidebar.slider(
    "Time Signature",
    min_value=1,
    max_value=7,
    value=4,
    step=1
)

# =========================
# VECTOR DE FEATURES
# =========================

features = np.array([[
    valence,
    energy,
    acousticness,
    danceability,
    tempo,
    loudness,
    speechiness,
    instrumentalness,
    liveness,
    duration_ms,
    explicit,
    time_signature
]])

# =========================
# PREDICCIÓN
# =========================

if st.button("🔮 Predecir"):

    features_scaled = scaler.transform(features)

    pred_genre = rf_genre.predict(features_scaled)
    pred_mood  = rf_mood.predict(features_scaled)

    genre = le_genre.inverse_transform(pred_genre)[0]
    mood  = le_mood.inverse_transform(pred_mood)[0]

    st.success("Predicción completada")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("🎵 Género", genre)

    with col2:
        st.metric("💭 Mood", mood)

    st.divider()

    st.subheader("📊 Datos ingresados")

    st.write({
        "valence": valence,
        "energy": energy,
        "acousticness": acousticness,
        "danceability": danceability,
        "tempo": tempo,
        "loudness": loudness,
        "speechiness": speechiness,
        "instrumentalness": instrumentalness,
        "liveness": liveness,
        "duration_ms": duration_ms,
        "explicit": explicit,
        "time_signature": time_signature
    })
