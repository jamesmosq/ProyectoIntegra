
"""
Aplicación Streamlit: Predictor de riesgo de deserción estudiantil
Universidad Santo Tomás - Maestría en Ciencia de Datos
Autor: James Mosquera Rentería
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Deserción",
    page_icon="graduation_cap",
    layout="wide"
)

# Título principal
st.title("Sistema de alerta temprana de deserción estudiantil")
st.markdown("---")

# Cargar modelo y scaler
@st.cache_resource
def cargar_modelo():
    modelo = joblib.load('modelo_desercion_final.pkl')
    scaler = joblib.load('scaler_desercion.pkl')
    return modelo, scaler

try:
    modelo, scaler = cargar_modelo()
    st.success("Modelo cargado exitosamente")
except:
    st.error("Error al cargar el modelo. Asegúrese de ejecutar primero el notebook.")
    st.stop()

# Sidebar con información
st.sidebar.header("Acerca del sistema")
st.sidebar.info(
    """
    Este sistema utiliza Machine Learning para predecir el riesgo 
    de deserción de estudiantes basándose en factores académicos,
    socioeconómicos y demográficos.

    **Métricas del modelo:**
    - Recall: ~70-80%
    - Precision: ~60-70%
    - F1-Score: ~65-75%
    """
)

# Formulario de entrada
st.header("Ingrese los datos del estudiante")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Datos personales")
    edad = st.number_input("Edad al matricularse", min_value=17, max_value=70, value=20)
    genero = st.selectbox("Género", options=[0, 1], format_func=lambda x: "Femenino" if x == 0 else "Masculino")
    estado_civil = st.selectbox("Estado civil", options=[1, 2, 3, 4, 5, 6], 
                                format_func=lambda x: {1:"Soltero", 2:"Casado", 3:"Viudo", 
                                                       4:"Divorciado", 5:"Unión libre", 6:"Separado"}[x])

with col2:
    st.subheader("Situación económica")
    becario = st.selectbox("¿Es becario?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
    deudor = st.selectbox("¿Tiene deudas con la institución?", options=[0, 1], 
                          format_func=lambda x: "No" if x == 0 else "Sí")
    matricula_dia = st.selectbox("¿Matrícula al día?", options=[0, 1], 
                                  format_func=lambda x: "No" if x == 0 else "Sí")

with col3:
    st.subheader("Rendimiento académico")
    materias_aprobadas_s1 = st.number_input("Materias aprobadas (1er semestre)", min_value=0, max_value=20, value=5)
    promedio_s1 = st.number_input("Promedio (1er semestre)", min_value=0.0, max_value=20.0, value=12.0, step=0.5)
    nota_admision = st.number_input("Nota de admisión", min_value=0.0, max_value=200.0, value=130.0)

st.markdown("---")

# Botón de predicción
if st.button("Realizar predicción", type="primary"):
    # Crear vector de características (simplificado para demo)
    datos_estudiante = np.zeros(36)

    # Asignar valores conocidos
    datos_estudiante[0] = estado_civil
    datos_estudiante[17] = genero
    datos_estudiante[18] = becario
    datos_estudiante[19] = edad
    datos_estudiante[15] = deudor
    datos_estudiante[16] = matricula_dia
    datos_estudiante[12] = nota_admision
    datos_estudiante[24] = materias_aprobadas_s1
    datos_estudiante[25] = promedio_s1

    # Escalar y predecir
    datos_scaled = scaler.transform([datos_estudiante])
    prediccion = modelo.predict(datos_scaled)[0]
    probabilidad = modelo.predict_proba(datos_scaled)[0]

    # Mostrar resultado
    st.header("Resultado de la predicción")

    col_res1, col_res2 = st.columns(2)

    with col_res1:
        if prediccion == 1:
            st.error("ALTO RIESGO DE DESERCIÓN")
            st.metric("Probabilidad de deserción", f"{probabilidad[1]*100:.1f}%")
        else:
            st.success("BAJO RIESGO DE DESERCIÓN")
            st.metric("Probabilidad de permanencia", f"{probabilidad[0]*100:.1f}%")

    with col_res2:
        st.subheader("Recomendaciones")
        if prediccion == 1:
            st.markdown("""
            **Acciones sugeridas:**
            - Asignar tutor académico
            - Evaluar apoyo financiero
            - Incluir en programa de mentoría
            - Realizar seguimiento semanal
            """)
        else:
            st.markdown("""
            **Acciones sugeridas:**
            - Mantener monitoreo regular
            - Ofrecer oportunidades de liderazgo
            - Incluir en estadísticas de éxito
            """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Ing James Mosquera Rentería | Maestría en ciencia de datos | Universidad Santo Tomás</p>
    </div>
    """,
    unsafe_allow_html=True
)
