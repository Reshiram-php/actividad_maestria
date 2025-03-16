import streamlit as st

perfiles = {
    "María (Estudiante)": ["Curso de Python", "Machine Learning Básico"],
    "Carlos (Profesional)": ["Deep Learning con TensorFlow", "IA para Negocios"],
    "Ana (Autodidacta)": ["Photoshop desde cero", "Ilustración digital"],
    "Luis (Docente)": ["Uso de tecnología en la enseñanza", "Gamificación"],
    "Fernando (Emprendedor)": ["Marketing en redes sociales", "SEO para principiantes"],
    "Gabriela (Madre trabajadora)": ["Fundamentos de programación", "Desarrollo web"],
    "Andrés (Recién graduado)": ["AutoCAD avanzado", "Gestión de proyectos"]
}

st.title("Recomendador de Cursos")
perfil = st.selectbox("Selecciona tu perfil de usuario", list(perfiles.keys()))

if st.button("Ver cursos recomendados"):
    st.write("### Cursos recomendados:")
    for curso in perfiles[perfil]:
        st.write(f"- {curso}")