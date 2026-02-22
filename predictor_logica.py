"""
Módulo de lógica de negocio para el predictor de deserción estudiantil.

Separa la lógica de predicción de la interfaz Streamlit para permitir
pruebas unitarias e integradas de forma independiente.

Universidad Santo Tomás - Maestría en Ciencia de Datos
Autor: James Mosquera Rentería
"""

from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import joblib


@dataclass
class DatosEstudiante:
    """Estructura de datos que representa un estudiante para la predicción."""
    edad: int
    genero: int           # 0=Femenino, 1=Masculino
    estado_civil: int     # 1=Soltero, 2=Casado, 3=Viudo, 4=Divorciado, 5=Unión libre, 6=Separado
    becario: int          # 0=No, 1=Sí
    deudor: int           # 0=No, 1=Sí
    matricula_dia: int    # 0=No, 1=Sí
    materias_aprobadas_s1: int
    promedio_s1: float
    nota_admision: float


@dataclass
class ResultadoPrediccion:
    """Estructura de datos para el resultado de la predicción de deserción."""
    es_alto_riesgo: bool
    probabilidad_desercion: float
    probabilidad_permanencia: float
    recomendaciones: List[str]


def validar_entrada(datos: DatosEstudiante) -> Tuple[bool, str]:
    """
    Valida que los datos de entrada del estudiante estén dentro de rangos válidos.

    Args:
        datos: Instancia de DatosEstudiante con los valores a validar.

    Returns:
        Tuple[bool, str]: (es_valido, mensaje_error).
                         Si es_valido=True el mensaje estará vacío.
    """
    if not (17 <= datos.edad <= 70):
        return False, "La edad debe estar entre 17 y 70 años"

    if datos.genero not in [0, 1]:
        return False, "El género debe ser 0 (Femenino) o 1 (Masculino)"

    if datos.estado_civil not in [1, 2, 3, 4, 5, 6]:
        return False, "El estado civil debe ser un valor entre 1 y 6"

    if datos.becario not in [0, 1]:
        return False, "El campo becario debe ser 0 o 1"

    if datos.deudor not in [0, 1]:
        return False, "El campo deudor debe ser 0 o 1"

    if datos.matricula_dia not in [0, 1]:
        return False, "El campo matrícula al día debe ser 0 o 1"

    if not (0 <= datos.materias_aprobadas_s1 <= 20):
        return False, "Las materias aprobadas deben estar entre 0 y 20"

    if not (0.0 <= datos.promedio_s1 <= 20.0):
        return False, "El promedio del primer semestre debe estar entre 0 y 20"

    if not (0.0 <= datos.nota_admision <= 200.0):
        return False, "La nota de admisión debe estar entre 0 y 200"

    return True, ""


def construir_vector_caracteristicas(datos: DatosEstudiante) -> np.ndarray:
    """
    Construye el vector de 36 características compatible con el modelo entrenado.

    Las posiciones no asignadas se inicializan en 0.0 (valor neutro).

    Args:
        datos: Instancia de DatosEstudiante con los valores del estudiante.

    Returns:
        np.ndarray: Vector de 36 elementos (dtype float64).
    """
    vector = np.zeros(36, dtype=np.float64)

    vector[0] = datos.estado_civil
    vector[12] = datos.nota_admision
    vector[15] = datos.deudor
    vector[16] = datos.matricula_dia
    vector[17] = datos.genero
    vector[18] = datos.becario
    vector[19] = datos.edad
    vector[24] = datos.materias_aprobadas_s1
    vector[25] = datos.promedio_s1

    return vector


def normalizar_datos(vector: np.ndarray, scaler) -> np.ndarray:
    """
    Normaliza el vector de características usando el StandardScaler entrenado.

    Args:
        vector: Array 1-D de 36 características sin normalizar.
        scaler: StandardScaler de sklearn ajustado durante el entrenamiento.

    Returns:
        np.ndarray: Array 2-D de shape (1, 36) con los datos normalizados.
    """
    return scaler.transform([vector])


def predecir_riesgo(vector_normalizado: np.ndarray, modelo) -> Tuple[int, np.ndarray]:
    """
    Realiza la predicción de riesgo de deserción sobre el vector normalizado.

    Args:
        vector_normalizado: Array 2-D de shape (1, 36) ya normalizado.
        modelo: Clasificador de sklearn entrenado con método predict y predict_proba.

    Returns:
        Tuple[int, np.ndarray]:
            - prediccion (int): 1 = Alto riesgo, 0 = Bajo riesgo.
            - probabilidades (np.ndarray): [prob_bajo_riesgo, prob_alto_riesgo].
    """
    prediccion = modelo.predict(vector_normalizado)[0]
    probabilidades = modelo.predict_proba(vector_normalizado)[0]
    return int(prediccion), probabilidades


def obtener_recomendaciones(es_alto_riesgo: bool) -> List[str]:
    """
    Genera lista de recomendaciones de intervención según el nivel de riesgo.

    Args:
        es_alto_riesgo: True si el estudiante tiene alto riesgo de deserción.

    Returns:
        List[str]: Lista de recomendaciones de acción.
    """
    if es_alto_riesgo:
        return [
            "Asignar tutor académico",
            "Evaluar apoyo financiero",
            "Incluir en programa de mentoría",
            "Realizar seguimiento semanal",
        ]
    return [
        "Mantener monitoreo regular",
        "Ofrecer oportunidades de liderazgo",
        "Incluir en estadísticas de éxito",
    ]


def cargar_modelos(
    ruta_modelo: str = "modelo_desercion_final.pkl",
    ruta_scaler: str = "scaler_desercion.pkl",
) -> Tuple[object, object]:
    """
    Carga el modelo de clasificación y el scaler desde archivos serializados.

    Args:
        ruta_modelo: Ruta al archivo .pkl del modelo entrenado.
        ruta_scaler: Ruta al archivo .pkl del StandardScaler.

    Returns:
        Tuple[modelo, scaler].

    Raises:
        FileNotFoundError: Si alguno de los archivos no existe.
    """
    modelo = joblib.load(ruta_modelo)
    scaler = joblib.load(ruta_scaler)
    return modelo, scaler


def ejecutar_prediccion_completa(
    datos: DatosEstudiante,
    modelo,
    scaler,
) -> ResultadoPrediccion:
    """
    Ejecuta el pipeline completo: validar → construir vector → normalizar → predecir.

    Args:
        datos: Datos del estudiante a evaluar.
        modelo: Clasificador de sklearn entrenado.
        scaler: StandardScaler de sklearn ajustado.

    Returns:
        ResultadoPrediccion con todos los detalles del resultado.

    Raises:
        ValueError: Si los datos de entrada no son válidos.
    """
    es_valido, mensaje_error = validar_entrada(datos)
    if not es_valido:
        raise ValueError(f"Datos de entrada inválidos: {mensaje_error}")

    vector = construir_vector_caracteristicas(datos)
    vector_normalizado = normalizar_datos(vector, scaler)
    prediccion, probabilidades = predecir_riesgo(vector_normalizado, modelo)

    es_alto_riesgo = prediccion == 1
    recomendaciones = obtener_recomendaciones(es_alto_riesgo)

    return ResultadoPrediccion(
        es_alto_riesgo=es_alto_riesgo,
        probabilidad_desercion=float(probabilidades[1]),
        probabilidad_permanencia=float(probabilidades[0]),
        recomendaciones=recomendaciones,
    )
