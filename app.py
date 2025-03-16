import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# ---- BASE DE DATOS DE CURSOS ----
cursos = pd.DataFrame({
    "id": list(range(1, 21)),  
    "nombre": [
        "Python Básico", "Machine Learning", "Deep Learning", "Photoshop", "SEO Avanzado",
        "Desarrollo Web con Django", "Excel para Negocios", "Marketing Digital", 
        "Edición de Video con Premiere", "SQL y Bases de Datos", "JavaScript Avanzado",
        "Ciberseguridad para Empresas", "Blockchain y Criptomonedas", "Diseño UX/UI",
        "Big Data y Analítica", "Análisis de Datos con Python", "Marketing de Contenidos",
        "Programación en C++", "Gestión de Proyectos Ágil", "Estrategias de Negociación"
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
        "Uso de Big Data para la toma de decisiones empresariales.",
        "Procesamiento de datos y visualización con Python.",
        "Cómo crear contenido de calidad para el marketing digital.",
        "Fundamentos de programación en C++.",
        "Gestión de proyectos con metodologías ágiles.",
        "Técnicas avanzadas de negociación y cierre de acuerdos."
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

# ---- GENERAMOS RATINGS FICTICIOS ----
ratings_data = []
for usuario, cursos_vistos in usuarios.items():
    for curso_id in cursos_vistos:
        rating = np.random.randint(3, 6)  # Calificaciones entre 3 y 5
        ratings_data.append([usuario, curso_id, rating])

ratings_df = pd.DataFrame(ratings_data, columns=["usuario", "curso_id", "rating"])

# ---- MATRIZ USUARIO-CURSO PARA SIMILARIDAD ----
matriz_usuarios = ratings_df.pivot(index="usuario", columns="curso_id", values="rating").fillna(0)

# ---- NORMALIZAR LOS DATOS ----
scaler = StandardScaler()
matriz_normalizada = scaler.fit_transform(matriz_usuarios)

# ---- CALCULAR SIMILARIDAD ENTRE USUARIOS ----
similitud_usuarios = cosine_similarity(matriz_normalizada)
usuarios_indices = {usuario: i for i, usuario in enumerate(matriz_usuarios.index)}

# ---- FUNCIÓN PARA RECOMENDAR CURSOS FIJOS ----
def recomendar_cursos(usuario_seleccionado):
    if usuario_seleccionado not in usuarios_indices:
        return []

    idx_usuario = usuarios_indices[usuario_seleccionado]
    similitudes = similitud_usuarios[idx_usuario]

    # Usuarios más similares
    indices_similares = np.argsort(similitudes)[::-1][1:3]
    usuarios_similares = [list(usuarios_indices.keys())[i] for i in indices_similares]

    # Cursos que el usuario no ha visto
    cursos_usuario_actual = set(ratings_df[ratings_df["usuario"] == usuario_seleccionado]["curso_id"])
    cursos_recomendados = set()

    for usuario_similar in usuarios_similares:
        cursos_similares = set(ratings_df[ratings_df["usuario"] == usuario_similar]["curso_id"])
        cursos_recomendados.update(cursos_similares - cursos_usuario_actual)

    # Tomar solo 3 cursos recomendados fijos
    cursos_recomendados = list(cursos_recomendados)[:3]
    
    return [(curso_id, cursos[cursos["id"] == curso_id]["nombre"].values[0]) for curso_id in cursos_recomendados]

# ---- FUNCIÓN PARA CURSOS ALTERNATIVOS ----
def cursos_alternativos():
    return cursos.sample(3)[["id", "nombre"]].values.tolist()

# ---- INTERFAZ EN STREAMLIT ----
st.title("🎓 Sistema de Recomendación de Cursos")

usuario_seleccionado = st.selectbox("Selecciona tu perfil", list(usuarios.keys()))

if usuario_seleccionado:
    st.subheader("📌 Tus cursos recomendados (fijos):")
    recomendaciones = recomendar_cursos(usuario_seleccionado)

    for curso_id, curso_nombre in recomendaciones:
        if st.button(f"✅ {curso_nombre}"):
            descripcion = cursos[cursos["id"] == curso_id]["descripcion"].values[0]
            st.write(f"📖 **Descripción:** {descripcion}")

    # Sección de cursos alternativos
    st.subheader("🔄 Cursos alternativos:")
    if st.button("Ver cursos alternativos"):
        nuevos_cursos = cursos_alternativos()
        for curso_id, curso_nombre in nuevos_cursos:
            st.write(f"📌 {curso_nombre}")
