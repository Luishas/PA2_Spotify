import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Configuración de la página ────────────────────────────────────────────────
st.set_page_config(
    page_title="Spotify ML App",
    page_icon="🎵",
    layout="centered"
)

# ── Carga de modelos (se cachea para no recargar cada vez) ────────────────────
@st.cache_resource
def cargar_modelos():
    rf_genre  = joblib.load('modelos/rf_genre.pkl')
    rf_mood   = joblib.load('modelos/rf_mood.pkl')
    scaler    = joblib.load('modelos/scaler.pkl')
    le_genre  = joblib.load('modelos/le_genre.pkl')
    le_mood   = joblib.load('modelos/le_mood.pkl')
    X_all_sc  = np.load('modelos/X_all_sc.npy')
    df_nombres  = pd.read_csv('modelos/df_nombres.csv')
    df_encoded  = pd.read_csv('modelos/df_encoded.csv')
    return rf_genre, rf_mood, scaler, le_genre, le_mood, X_all_sc, df_nombres, df_encoded

rf_genre, rf_mood, scaler, le_genre, le_mood, X_all_sc, df_nombres, df_encoded = cargar_modelos()

# ── Título ────────────────────────────────────────────────────────────────────
st.title("🎵 Spotify — Predicción y Recomendación")
st.markdown("Ingresa las características de audio de una canción y te digo su género, su mood y canciones similares.")
st.divider()

# ── Inputs del usuario ────────────────────────────────────────────────────────
st.subheader("Características de audio")

col1, col2, col3 = st.columns(3)

with col1:
    valence        = st.slider("Valence",        0.0, 1.0, 0.5, 0.01, help="Positividad emocional (0=triste, 1=feliz)")
    energy         = st.slider("Energy",         0.0, 1.0, 0.5, 0.01, help="Intensidad de la canción")
    acousticness   = st.slider("Acousticness",   0.0, 1.0, 0.3, 0.01, help="Probabilidad de ser acústica")
    danceability   = st.slider("Danceability",   0.0, 1.0, 0.6, 0.01)

with col2:
    speechiness    = st.slider("Speechiness",    0.0, 1.0, 0.05, 0.01)
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0, 0.01)
    liveness       = st.slider("Liveness",       0.0, 1.0, 0.1, 0.01)
    explicit       = st.selectbox("Explicit", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")

with col3:
    tempo          = st.number_input("Tempo (BPM)",     min_value=50.0,   max_value=250.0,  value=120.0)
    loudness       = st.number_input("Loudness (dB)",   min_value=-60.0,  max_value=0.0,    value=-8.0)
    duration_ms    = st.number_input("Duración (ms)",   min_value=30000,  max_value=600000, value=200000)
    time_signature = st.selectbox("Time Signature", [1, 2, 3, 4, 5], index=3)

# ── Predicción ────────────────────────────────────────────────────────────────
st.divider()

if st.button("🎯 Predecir", use_container_width=True, type="primary"):

    entrada = np.array([[valence, energy, acousticness, danceability,
                         tempo, loudness, speechiness, instrumentalness,
                         liveness, duration_ms, explicit, time_signature]])

    entrada_sc = scaler.transform(entrada)

    genero_pred = le_genre.inverse_transform(rf_genre.predict(entrada_sc))[0]
    mood_pred   = le_mood.inverse_transform(rf_mood.predict(entrada_sc))[0]

    # Probabilidades
    prob_genre = rf_genre.predict_proba(entrada_sc)[0]
    prob_mood  = rf_mood.predict_proba(entrada_sc)[0]

    conf_genre = prob_genre.max() * 100
    conf_mood  = prob_mood.max() * 100

    # Emojis por mood
    mood_emoji = {
        'happy': '😄', 'sad': '😢', 'energetic': '⚡',
        'romantic': '💕', 'dark': '🌑', 'calm': '😌'
    }

    # Resultados
    st.subheader("Resultados")
    col_r1, col_r2 = st.columns(2)

    with col_r1:
        st.metric(
            label="🎸 Género predicho",
            value=genero_pred.capitalize(),
            delta=f"Confianza: {conf_genre:.1f}%"
        )

    with col_r2:
        emoji = mood_emoji.get(mood_pred, '🎵')
        st.metric(
            label=f"{emoji} Mood predicho",
            value=mood_pred.capitalize(),
            delta=f"Confianza: {conf_mood:.1f}%"
        )

    # ── Recomendaciones ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("🎧 Canciones similares")

    genero_enc = rf_genre.predict(entrada_sc)[0]
    mood_enc   = rf_mood.predict(entrada_sc)[0]

    mask = (
        (df_encoded['genre_encoded'] == genero_enc) &
        (df_encoded['mood_encoded']  == mood_enc)
    )
    posiciones = df_encoded[mask].index.tolist()

    if len(posiciones) < 2:
        st.warning("No hay suficientes canciones similares con ese género y mood.")
    else:
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(entrada_sc, X_all_sc[posiciones])[0]
        top5 = np.argsort(sims)[::-1][:5]

        for rank, i in enumerate(top5, 1):
            pos_real = posiciones[i]
            nombre = df_nombres.iloc[pos_real]
            similitud = sims[i] * 100

            with st.container():
                st.markdown(f"**{rank}. {nombre['track_name']}** — {nombre['artists']}")
                st.progress(similitud / 100, text=f"Similitud: {similitud:.1f}%")

    # ── Distribución de probabilidades por género ─────────────────────────────
    with st.expander("Ver probabilidades por género"):
        df_prob = pd.DataFrame({
            'Género': le_genre.classes_,
            'Probabilidad': prob_genre
        }).sort_values('Probabilidad', ascending=False)
        st.bar_chart(df_prob.set_index('Género')['Probabilidad'])

    with st.expander("Ver probabilidades por mood"):
        df_prob_m = pd.DataFrame({
            'Mood': le_mood.classes_,
            'Probabilidad': prob_mood
        }).sort_values('Probabilidad', ascending=False)
        st.bar_chart(df_prob_m.set_index('Mood')['Probabilidad'])
