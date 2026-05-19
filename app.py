import streamlit as st
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
