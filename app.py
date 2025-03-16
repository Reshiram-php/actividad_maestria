import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, KNNBasic, SVD
from surprise.model_selection import train_test_split

# ---- SEMILLA PARA RESULTADOS CONSISTENTES ----
np.random.seed(42)

# ---- NUEVA BASE DE DATOS DE CURSOS ----
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

# ---- GENERAMOS RATINGS FICTICIOS PARA FILTRADO COLABORATIVO ----
ratings_data = []
for usuario, cursos_vistos in usuarios.items():
    for curso_id in cursos_vistos:
        rating = np.random.randint(1, 6)  # Simulamos calificaciones entre 3 y 5
        ratings_data.append([usuario, curso_id, rating])

ratings_df = pd.DataFrame(ratings_data, columns=["usuario", "curso_id", "rating"])

# ---- CONFIGURAR FILTRADO COLABORATIVO CON SURPRISE ----
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['usuario', 'curso_id', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)

model = SVD()
model.fit(trainset)

# ---- FUNCIÓN PARA OBTENER RECOMENDACIONES ----
def recomendar_cursos(usuario_seleccionado):
    cursos_vistos = set(ratings_df[ratings_df["usuario"] == usuario_seleccionado]["curso_id"])
    cursos_no_vistos = set(cursos["id"]) - cursos_vistos
    
    predicciones = []
    for curso_id in cursos_no_vistos:
        pred = model.predict(usuario_seleccionado, curso_id)
        predicciones.append((curso_id, pred.est))
    
    # Ordenamos por la calificación estimada
    predicciones.sort(key=lambda x: x[1], reverse=True)
    
    # Mostramos las 3 mejores recomendaciones
    cursos_recomendados = [(curso_id, cursos[cursos["id"] == curso_id]["nombre"].values[0]) for curso_id, _ in predicciones[:3]]
    return cursos_recomendados

# ---- INTERFAZ EN STREAMLIT ----
st.title("🎓 Sistema de Recomendación de Cursos")

usuario_seleccionado = st.selectbox("Selecciona tu perfil", list(usuarios.keys()))

if usuario_seleccionado:
    st.subheader("📌 Tus cursos recomendados:")
    recomendaciones = recomendar_cursos(usuario_seleccionado)

    for curso_id, curso_nombre in recomendaciones:
        if st.button(f"✅ {curso_nombre}"):
            descripcion = cursos[cursos["id"] == curso_id]["descripcion"].values[0]
            st.write(f"📖 **Descripción:** {descripcion}")
