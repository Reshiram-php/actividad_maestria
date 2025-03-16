import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# ---- BASE DE DATOS DE CURSOS ----
cursos = pd.DataFrame({
    "id": list(range(1, 16)),  # IDs del 1 al 15
    "nombre": [
        "Python Básico", "Machine Learning", "Deep Learning", "Photoshop", "SEO Avanzado",
        "Desarrollo Web con Django", "Excel para Negocios", "Marketing Digital", 
        "Edición de Video con Premiere", "SQL y Bases de Datos", "JavaScript Avanzado",
        "Ciberseguridad para Empresas", "Blockchain y Criptomonedas", "Diseño UX/UI",
        "Big Data y Analítica"
    ],
    "descripcion": [
        "Curso para aprender Python desde cero.",
        "Introducción al Machine Learning con Python.",
        "Redes neuronales profundas y TensorFlow.",
        "Edición de imágenes con Photoshop.",
        "Técnicas avanzadas de optimización SEO.",
        "Construcción de aplicaciones web con Django y Python.",
        "Análisis de datos y automatización con Excel.",
        "Estrategias efectivas de marketing digital y redes sociales.",
        "Creación y edición profesional de videos con Premiere Pro.",
        "Fundamentos de bases de datos y consultas SQL.",
        "JavaScript avanzado con frameworks modernos.",
        "Protección de datos y ciberseguridad empresarial.",
        "Introducción a blockchain y criptomonedas.",
        "Principios de diseño UX/UI para aplicaciones y web.",
        "Uso de Big Data para la toma de decisiones empresariales."
    ]
})

# ---- PERFIL DE USUARIOS Y CURSOS QUE LES INTERESAN ----
usuarios = {
    "María (Estudiante)": [1, 2, 6],
    "Carlos (Profesional)": [2, 3, 10, 15],
    "Ana (Autodidacta)": [4, 8, 14],
    "Luis (Docente)": [3, 4, 10, 12],
    "Fernando (Emprendedor)": [5, 8, 13],
    "Gabriela (Madre trabajadora)": [1, 7, 11],
    "Andrés (Recién graduado)": [1, 3, 10, 15]
}

# ---- INICIALIZAMOS EL ESTADO DE STREAMLIT ----
if "ratings_df" not in st.session_state:
    ratings_data = []
    np.random.seed(42)  # Semilla fija para evitar cambios aleatorios
    for usuario, cursos_vistos in usuarios.items():
        for curso_id in cursos_vistos:
            rating = np.random.randint(1, 6)  # Calificaciones entre 1 y 5
            ratings_data.append([usuario, curso_id, rating])
    
    st.session_state.ratings_df = pd.DataFrame(ratings_data, columns=["usuario", "curso_id", "rating"])

# ---- FUNCIÓN PARA CALCULAR RECOMENDACIONES ----
def recomendar_cursos(usuario_seleccionado):
    ratings_df = st.session_state.ratings_df.copy()

    # Generar matriz usuario-curso
    matriz_usuarios = ratings_df.pivot(index="usuario", columns="curso_id", values="rating").fillna(0)

    # Normalizar datos
    scaler = StandardScaler()
    matriz_normalizada = scaler.fit_transform(matriz_usuarios)

    # Similitud entre usuarios
    similitud_usuarios = cosine_similarity(matriz_normalizada)
    usuarios_indices = {usuario: i for i, usuario in enumerate(matriz_usuarios.index)}

    if usuario_seleccionado not in usuarios_indices:
        return []

    idx_usuario = usuarios_indices[usuario_seleccionado]
    similitudes = similitud_usuarios[idx_usuario]

    # Buscar usuarios similares
    indices_similares = np.argsort(similitudes)[::-1][1:3]
    usuarios_similares = [list(usuarios_indices.keys())[i] for i in indices_similares]

    # Obtener los cursos que estos usuarios han visto y que el usuario actual no ha visto
    cursos_usuario_actual = set(ratings_df[ratings_df["usuario"] == usuario_seleccionado]["curso_id"])
    cursos_recomendados = set()

    for usuario_similar in usuarios_similares:
        cursos_similares = set(ratings_df[ratings_df["usuario"] == usuario_similar]["curso_id"])
        cursos_recomendados.update(cursos_similares - cursos_usuario_actual)

    # Tomar solo 3 cursos recomendados
    cursos_recomendados = list(cursos_recomendados)[:3]
    
    # Convertir IDs en nombres de cursos
    return [(curso_id, cursos[cursos["id"] == curso_id]["nombre"].values[0]) for curso_id in cursos_recomendados]

# ---- INTERFAZ STREAMLIT ----
st.title("🎓 Sistema de Recomendación de Cursos")

usuario_seleccionado = st.selectbox("Selecciona tu perfil", list(usuarios.keys()))

if usuario_seleccionado:
    st.subheader("📌 Tus cursos recomendados:")
    recomendaciones = recomendar_cursos(usuario_seleccionado)

    for curso_id, curso_nombre in recomendaciones:
        if st.button(f"✅ {curso_nombre}", key=f"rec_{curso_id}"):
            descripcion = cursos[cursos["id"] == curso_id]["descripcion"].values[0]
            st.write(f"📖 **Descripción:** {descripcion}")

    # ---- SECCIÓN DE CALIFICACIÓN ----
    st.subheader("⭐ Califica un curso")

    # Mostrar lista de cursos disponibles para calificar
    curso_para_calificar = st.selectbox("Selecciona un curso", cursos["nombre"])
    curso_id = cursos[cursos["nombre"] == curso_para_calificar]["id"].values[0]
    nueva_calificacion = st.slider("Elige tu calificación (1-5)", 1, 5, 3)

    if st.button("Enviar Calificación", key=f"rate_{curso_id}"):
        # Actualizar o agregar la calificación en el DataFrame
        if ((st.session_state.ratings_df["usuario"] == usuario_seleccionado) & 
            (st.session_state.ratings_df["curso_id"] == curso_id)).any():
            st.session_state.ratings_df.loc[
                (st.session_state.ratings_df["usuario"] == usuario_seleccionado) & 
                (st.session_state.ratings_df["curso_id"] == curso_id), 
                "rating"
            ] = nueva_calificacion
        else:
            nueva_fila = pd.DataFrame([[usuario_seleccionado, curso_id, nueva_calificacion]], 
                                      columns=["usuario", "curso_id", "rating"])
            st.session_state.ratings_df = pd.concat([st.session_state.ratings_df, nueva_fila], ignore_index=True)

        st.rerun()

    # ---- HISTORIAL DE CALIFICACIONES ----
    st.subheader("📜 Historial de Calificaciones")
    historial_usuario = st.session_state.ratings_df[st.session_state.ratings_df["usuario"] == usuario_seleccionado]
    
    if not historial_usuario.empty:
        for _, row in historial_usuario.iterrows():
            curso_nombre = cursos[cursos["id"] == row["curso_id"]]["nombre"].values[0]
            st.write(f"📘 **{curso_nombre}** - ⭐ {row['rating']}/5")

