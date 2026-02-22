"""
Pruebas unitarias para el módulo predictor_logica.py

Casos de prueba identificados:
  U-01 a U-20: validar_entrada
  U-21 a U-32: construir_vector_caracteristicas
  U-33 a U-35: normalizar_datos
  U-36 a U-40: predecir_riesgo
  U-41 a U-48: obtener_recomendaciones
  U-49 a U-51: cargar_modelos
  U-52 a U-57: ejecutar_prediccion_completa

Universidad Santo Tomás - Maestría en Ciencia de Datos
Autor: James Mosquera Rentería
"""

import sys
import os
import unittest
from dataclasses import replace
from unittest.mock import MagicMock, patch

import numpy as np

# Agregar directorio raíz del proyecto al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictor_logica import (
    DatosEstudiante,
    ResultadoPrediccion,
    validar_entrada,
    construir_vector_caracteristicas,
    normalizar_datos,
    predecir_riesgo,
    obtener_recomendaciones,
    cargar_modelos,
    ejecutar_prediccion_completa,
)


# ---------------------------------------------------------------------------
# Fixture base reutilizable
# ---------------------------------------------------------------------------
DATOS_BASE = DatosEstudiante(
    edad=20,
    genero=1,
    estado_civil=1,
    becario=0,
    deudor=0,
    matricula_dia=1,
    materias_aprobadas_s1=5,
    promedio_s1=12.0,
    nota_admision=130.0,
)


# ===========================================================================
# SECCIÓN 1: validar_entrada  (U-01 – U-20)
# ===========================================================================
class TestValidarEntrada(unittest.TestCase):
    """Pruebas unitarias para validar_entrada."""

    # --- Caso base ---
    def test_U01_datos_validos_devuelve_true(self):
        """U-01: Datos completamente válidos devuelven (True, '')."""
        es_valido, mensaje = validar_entrada(DATOS_BASE)
        self.assertTrue(es_valido)
        self.assertEqual(mensaje, "")

    # --- Edad ---
    def test_U02_edad_minima_17_valida(self):
        """U-02: Edad mínima (17) es aceptada."""
        es_valido, _ = validar_entrada(replace(DATOS_BASE, edad=17))
        self.assertTrue(es_valido)

    def test_U03_edad_maxima_70_valida(self):
        """U-03: Edad máxima (70) es aceptada."""
        es_valido, _ = validar_entrada(replace(DATOS_BASE, edad=70))
        self.assertTrue(es_valido)

    def test_U04_edad_menor_17_invalida(self):
        """U-04: Edad 16 es rechazada con mensaje descriptivo."""
        es_valido, mensaje = validar_entrada(replace(DATOS_BASE, edad=16))
        self.assertFalse(es_valido)
        self.assertIn("edad", mensaje.lower())

    def test_U05_edad_mayor_70_invalida(self):
        """U-05: Edad 71 es rechazada con mensaje descriptivo."""
        es_valido, mensaje = validar_entrada(replace(DATOS_BASE, edad=71))
        self.assertFalse(es_valido)
        self.assertIn("edad", mensaje.lower())

    # --- Género ---
    def test_U06_genero_0_femenino_valido(self):
        """U-06: Género 0 (Femenino) es aceptado."""
        es_valido, _ = validar_entrada(replace(DATOS_BASE, genero=0))
        self.assertTrue(es_valido)

    def test_U07_genero_1_masculino_valido(self):
        """U-07: Género 1 (Masculino) es aceptado."""
        es_valido, _ = validar_entrada(replace(DATOS_BASE, genero=1))
        self.assertTrue(es_valido)

    def test_U08_genero_2_invalido(self):
        """U-08: Género 2 (valor no definido) es rechazado."""
        es_valido, mensaje = validar_entrada(replace(DATOS_BASE, genero=2))
        self.assertFalse(es_valido)
        self.assertIn("género", mensaje.lower())

    # --- Estado civil ---
    def test_U09_todos_estados_civiles_1_a_6_validos(self):
        """U-09: Estado civil 1-6 son todos aceptados."""
        for estado in range(1, 7):
            with self.subTest(estado_civil=estado):
                es_valido, _ = validar_entrada(replace(DATOS_BASE, estado_civil=estado))
                self.assertTrue(es_valido)

    def test_U10_estado_civil_0_invalido(self):
        """U-10: Estado civil 0 (fuera de rango) es rechazado."""
        es_valido, _ = validar_entrada(replace(DATOS_BASE, estado_civil=0))
        self.assertFalse(es_valido)

    def test_U11_estado_civil_7_invalido(self):
        """U-11: Estado civil 7 (fuera de rango) es rechazado."""
        es_valido, _ = validar_entrada(replace(DATOS_BASE, estado_civil=7))
        self.assertFalse(es_valido)

    # --- Promedio semestre 1 ---
    def test_U12_promedio_s1_cero_valido(self):
        """U-12: Promedio 0.0 es aceptado."""
        es_valido, _ = validar_entrada(replace(DATOS_BASE, promedio_s1=0.0))
        self.assertTrue(es_valido)

    def test_U13_promedio_s1_maximo_20_valido(self):
        """U-13: Promedio 20.0 es aceptado."""
        es_valido, _ = validar_entrada(replace(DATOS_BASE, promedio_s1=20.0))
        self.assertTrue(es_valido)

    def test_U14_promedio_s1_negativo_invalido(self):
        """U-14: Promedio -0.1 es rechazado con mensaje descriptivo."""
        es_valido, mensaje = validar_entrada(replace(DATOS_BASE, promedio_s1=-0.1))
        self.assertFalse(es_valido)
        self.assertIn("promedio", mensaje.lower())

    def test_U15_promedio_s1_mayor_20_invalido(self):
        """U-15: Promedio 20.1 es rechazado."""
        es_valido, _ = validar_entrada(replace(DATOS_BASE, promedio_s1=20.1))
        self.assertFalse(es_valido)

    # --- Nota de admisión ---
    def test_U16_nota_admision_minima_0_valida(self):
        """U-16: Nota de admisión 0.0 es aceptada."""
        es_valido, _ = validar_entrada(replace(DATOS_BASE, nota_admision=0.0))
        self.assertTrue(es_valido)

    def test_U17_nota_admision_maxima_200_valida(self):
        """U-17: Nota de admisión 200.0 es aceptada."""
        es_valido, _ = validar_entrada(replace(DATOS_BASE, nota_admision=200.0))
        self.assertTrue(es_valido)

    def test_U18_nota_admision_201_invalida(self):
        """U-18: Nota de admisión 201 es rechazada con mensaje descriptivo."""
        es_valido, mensaje = validar_entrada(replace(DATOS_BASE, nota_admision=201.0))
        self.assertFalse(es_valido)
        self.assertIn("admisión", mensaje.lower())

    # --- Materias aprobadas ---
    def test_U19_materias_aprobadas_0_valido(self):
        """U-19: 0 materias aprobadas es aceptado."""
        es_valido, _ = validar_entrada(replace(DATOS_BASE, materias_aprobadas_s1=0))
        self.assertTrue(es_valido)

    def test_U20_materias_aprobadas_negativas_invalido(self):
        """U-20: Materias aprobadas negativas son rechazadas."""
        es_valido, _ = validar_entrada(replace(DATOS_BASE, materias_aprobadas_s1=-1))
        self.assertFalse(es_valido)


# ===========================================================================
# SECCIÓN 2: construir_vector_caracteristicas  (U-21 – U-32)
# ===========================================================================
class TestConstruirVectorCaracteristicas(unittest.TestCase):
    """Pruebas unitarias para construir_vector_caracteristicas."""

    def setUp(self):
        self.vector = construir_vector_caracteristicas(DATOS_BASE)

    def test_U21_vector_tiene_36_elementos(self):
        """U-21: El vector debe tener exactamente 36 elementos."""
        self.assertEqual(len(self.vector), 36)

    def test_U22_vector_es_ndarray(self):
        """U-22: El resultado debe ser un np.ndarray."""
        self.assertIsInstance(self.vector, np.ndarray)

    def test_U23_vector_dtype_float64(self):
        """U-23: El dtype del vector debe ser float64."""
        self.assertEqual(self.vector.dtype, np.float64)

    def test_U24_estado_civil_en_posicion_0(self):
        """U-24: estado_civil se asigna en índice [0]."""
        self.assertEqual(self.vector[0], DATOS_BASE.estado_civil)

    def test_U25_nota_admision_en_posicion_12(self):
        """U-25: nota_admision se asigna en índice [12]."""
        self.assertEqual(self.vector[12], DATOS_BASE.nota_admision)

    def test_U26_deudor_en_posicion_15(self):
        """U-26: deudor se asigna en índice [15]."""
        self.assertEqual(self.vector[15], DATOS_BASE.deudor)

    def test_U27_matricula_dia_en_posicion_16(self):
        """U-27: matricula_dia se asigna en índice [16]."""
        self.assertEqual(self.vector[16], DATOS_BASE.matricula_dia)

    def test_U28_genero_en_posicion_17(self):
        """U-28: genero se asigna en índice [17]."""
        self.assertEqual(self.vector[17], DATOS_BASE.genero)

    def test_U29_becario_en_posicion_18(self):
        """U-29: becario se asigna en índice [18]."""
        self.assertEqual(self.vector[18], DATOS_BASE.becario)

    def test_U30_edad_en_posicion_19(self):
        """U-30: edad se asigna en índice [19]."""
        self.assertEqual(self.vector[19], DATOS_BASE.edad)

    def test_U31_materias_aprobadas_en_posicion_24(self):
        """U-31: materias_aprobadas_s1 se asigna en índice [24]."""
        self.assertEqual(self.vector[24], DATOS_BASE.materias_aprobadas_s1)

    def test_U32_posiciones_no_asignadas_son_cero(self):
        """U-32: Todas las posiciones no asignadas deben ser 0.0."""
        posiciones_asignadas = {0, 12, 15, 16, 17, 18, 19, 24, 25}
        for idx, valor in enumerate(self.vector):
            if idx not in posiciones_asignadas:
                self.assertEqual(
                    valor, 0.0,
                    f"Posición {idx} debería ser 0.0 pero es {valor}",
                )


# ===========================================================================
# SECCIÓN 3: normalizar_datos  (U-33 – U-35)
# ===========================================================================
class TestNormalizarDatos(unittest.TestCase):
    """Pruebas unitarias para normalizar_datos."""

    def _scaler_mock(self, shape=(1, 36)):
        scaler = MagicMock()
        scaler.transform.return_value = np.zeros(shape)
        return scaler

    def test_U33_resultado_shape_1x36(self):
        """U-33: La normalización devuelve un array de shape (1, 36)."""
        scaler = self._scaler_mock((1, 36))
        resultado = normalizar_datos(np.zeros(36), scaler)
        self.assertEqual(resultado.shape, (1, 36))

    def test_U34_llama_transform_una_vez(self):
        """U-34: Se invoca scaler.transform exactamente una vez."""
        scaler = self._scaler_mock()
        normalizar_datos(np.zeros(36), scaler)
        scaler.transform.assert_called_once()

    def test_U35_pasa_vector_correcto_al_scaler(self):
        """U-35: El vector entregado al scaler coincide con el vector de entrada."""
        scaler = self._scaler_mock()
        vector_entrada = np.arange(36, dtype=np.float64)
        normalizar_datos(vector_entrada, scaler)
        args_llamada = scaler.transform.call_args[0][0]
        np.testing.assert_array_equal(args_llamada[0], vector_entrada)


# ===========================================================================
# SECCIÓN 4: predecir_riesgo  (U-36 – U-40)
# ===========================================================================
class TestPredecirRiesgo(unittest.TestCase):
    """Pruebas unitarias para predecir_riesgo."""

    def _modelo_mock(self, clase=1, probs=(0.3, 0.7)):
        modelo = MagicMock()
        modelo.predict.return_value = np.array([clase])
        modelo.predict_proba.return_value = np.array([list(probs)])
        return modelo

    def test_U36_prediccion_alto_riesgo_devuelve_1(self):
        """U-36: Cuando el modelo predice 1, se retorna prediccion=1."""
        prediccion, _ = predecir_riesgo(np.zeros((1, 36)), self._modelo_mock(clase=1))
        self.assertEqual(prediccion, 1)

    def test_U37_prediccion_bajo_riesgo_devuelve_0(self):
        """U-37: Cuando el modelo predice 0, se retorna prediccion=0."""
        prediccion, _ = predecir_riesgo(
            np.zeros((1, 36)), self._modelo_mock(clase=0, probs=(0.8, 0.2))
        )
        self.assertEqual(prediccion, 0)

    def test_U38_probabilidades_tienen_dos_elementos(self):
        """U-38: El array de probabilidades tiene longitud 2."""
        _, probabilidades = predecir_riesgo(np.zeros((1, 36)), self._modelo_mock())
        self.assertEqual(len(probabilidades), 2)

    def test_U39_probabilidades_suman_1(self):
        """U-39: La suma de probabilidades es ~1.0."""
        _, probabilidades = predecir_riesgo(np.zeros((1, 36)), self._modelo_mock())
        self.assertAlmostEqual(float(sum(probabilidades)), 1.0, places=5)

    def test_U40_prediccion_es_int(self):
        """U-40: El valor de predicción retornado es de tipo int."""
        prediccion, _ = predecir_riesgo(np.zeros((1, 36)), self._modelo_mock())
        self.assertIsInstance(prediccion, int)


# ===========================================================================
# SECCIÓN 5: obtener_recomendaciones  (U-41 – U-48)
# ===========================================================================
class TestObtenerRecomendaciones(unittest.TestCase):
    """Pruebas unitarias para obtener_recomendaciones."""

    def test_U41_alto_riesgo_lista_no_vacia(self):
        """U-41: Alto riesgo devuelve al menos una recomendación."""
        rec = obtener_recomendaciones(es_alto_riesgo=True)
        self.assertGreater(len(rec), 0)

    def test_U42_bajo_riesgo_lista_no_vacia(self):
        """U-42: Bajo riesgo devuelve al menos una recomendación."""
        rec = obtener_recomendaciones(es_alto_riesgo=False)
        self.assertGreater(len(rec), 0)

    def test_U43_alto_riesgo_menciona_tutor(self):
        """U-43: Recomendaciones de alto riesgo incluyen asignación de tutor."""
        rec = obtener_recomendaciones(es_alto_riesgo=True)
        self.assertTrue(any("tutor" in r.lower() for r in rec))

    def test_U44_alto_riesgo_menciona_seguimiento(self):
        """U-44: Recomendaciones de alto riesgo incluyen seguimiento."""
        rec = obtener_recomendaciones(es_alto_riesgo=True)
        self.assertTrue(any("seguimiento" in r.lower() for r in rec))

    def test_U45_bajo_riesgo_menciona_monitoreo(self):
        """U-45: Recomendaciones de bajo riesgo incluyen monitoreo."""
        rec = obtener_recomendaciones(es_alto_riesgo=False)
        self.assertTrue(any("monitoreo" in r.lower() for r in rec))

    def test_U46_resultado_es_lista(self):
        """U-46: El retorno es siempre de tipo list."""
        self.assertIsInstance(obtener_recomendaciones(True), list)
        self.assertIsInstance(obtener_recomendaciones(False), list)

    def test_U47_elementos_son_strings(self):
        """U-47: Cada elemento de la lista es un str."""
        for rec in obtener_recomendaciones(True) + obtener_recomendaciones(False):
            with self.subTest(recomendacion=rec):
                self.assertIsInstance(rec, str)

    def test_U48_alto_riesgo_tiene_mas_o_igual_recomendaciones_que_bajo(self):
        """U-48: Alto riesgo tiene al menos tantas recomendaciones como bajo riesgo."""
        rec_alto = obtener_recomendaciones(True)
        rec_bajo = obtener_recomendaciones(False)
        self.assertGreaterEqual(len(rec_alto), len(rec_bajo))


# ===========================================================================
# SECCIÓN 6: cargar_modelos  (U-49 – U-51)
# ===========================================================================
class TestCargarModelos(unittest.TestCase):
    """Pruebas unitarias para cargar_modelos (con mock de joblib)."""

    @patch("predictor_logica.joblib.load")
    def test_U49_carga_exitosa_devuelve_modelo_y_scaler(self, mock_load):
        """U-49: Carga exitosa devuelve el modelo y el scaler en orden."""
        modelo_fake = MagicMock()
        scaler_fake = MagicMock()
        mock_load.side_effect = [modelo_fake, scaler_fake]

        modelo, scaler = cargar_modelos("modelo.pkl", "scaler.pkl")

        self.assertIs(modelo, modelo_fake)
        self.assertIs(scaler, scaler_fake)

    @patch("predictor_logica.joblib.load")
    def test_U50_filenotfounderror_se_propaga(self, mock_load):
        """U-50: FileNotFoundError lanzado por joblib se propaga al caller."""
        mock_load.side_effect = FileNotFoundError("Archivo no encontrado")

        with self.assertRaises(FileNotFoundError):
            cargar_modelos("no_existe.pkl", "tampoco.pkl")

    @patch("predictor_logica.joblib.load")
    def test_U51_joblib_load_se_llama_dos_veces(self, mock_load):
        """U-51: joblib.load se invoca exactamente dos veces (modelo + scaler)."""
        mock_load.return_value = MagicMock()
        cargar_modelos("modelo.pkl", "scaler.pkl")
        self.assertEqual(mock_load.call_count, 2)


# ===========================================================================
# SECCIÓN 7: ejecutar_prediccion_completa  (U-52 – U-57)
# ===========================================================================
class TestEjecutarPrediccionCompleta(unittest.TestCase):
    """Pruebas unitarias para ejecutar_prediccion_completa."""

    def setUp(self):
        self.modelo = MagicMock()
        self.modelo.predict.return_value = np.array([0])
        self.modelo.predict_proba.return_value = np.array([[0.75, 0.25]])

        self.scaler = MagicMock()
        self.scaler.transform.return_value = np.zeros((1, 36))

    def test_U52_resultado_es_tipo_ResultadoPrediccion(self):
        """U-52: El objeto retornado es de tipo ResultadoPrediccion."""
        resultado = ejecutar_prediccion_completa(DATOS_BASE, self.modelo, self.scaler)
        self.assertIsInstance(resultado, ResultadoPrediccion)

    def test_U53_alto_riesgo_se_refleja_en_resultado(self):
        """U-53: Prediccion 1 -> es_alto_riesgo=True en el resultado."""
        self.modelo.predict.return_value = np.array([1])
        self.modelo.predict_proba.return_value = np.array([[0.2, 0.8]])

        resultado = ejecutar_prediccion_completa(DATOS_BASE, self.modelo, self.scaler)

        self.assertTrue(resultado.es_alto_riesgo)

    def test_U54_bajo_riesgo_se_refleja_en_resultado(self):
        """U-54: Prediccion 0 -> es_alto_riesgo=False en el resultado."""
        resultado = ejecutar_prediccion_completa(DATOS_BASE, self.modelo, self.scaler)
        self.assertFalse(resultado.es_alto_riesgo)

    def test_U55_probabilidades_asignadas_correctamente(self):
        """U-55: probabilidad_permanencia y probabilidad_desercion se asignan."""
        resultado = ejecutar_prediccion_completa(DATOS_BASE, self.modelo, self.scaler)
        self.assertAlmostEqual(resultado.probabilidad_permanencia, 0.75, places=5)
        self.assertAlmostEqual(resultado.probabilidad_desercion, 0.25, places=5)

    def test_U56_datos_invalidos_lanzan_ValueError(self):
        """U-56: Datos con edad inválida lanzan ValueError antes de predecir."""
        datos_malos = replace(DATOS_BASE, edad=5)
        with self.assertRaises(ValueError):
            ejecutar_prediccion_completa(datos_malos, self.modelo, self.scaler)

    def test_U57_resultado_contiene_recomendaciones_no_vacias(self):
        """U-57: El resultado siempre incluye al menos una recomendación."""
        resultado = ejecutar_prediccion_completa(DATOS_BASE, self.modelo, self.scaler)
        self.assertIsNotNone(resultado.recomendaciones)
        self.assertGreater(len(resultado.recomendaciones), 0)


# ===========================================================================
# Punto de entrada
# ===========================================================================
if __name__ == "__main__":
    unittest.main(verbosity=2)
