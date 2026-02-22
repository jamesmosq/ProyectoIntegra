"""
Pruebas de integración para el sistema predictor de deserción estudiantil.

Casos de prueba identificados:
  I-01 a I-05: Carga de modelos reales (archivos .pkl)
  I-06 a I-10: Flujo completo con modelo y scaler reales
  I-11 a I-17: Coherencia con el dataset real (data.csv)
  I-18 a I-23: Pipeline de extremo a extremo con mocks

Los tests que requieren archivos reales (.pkl, .csv) se marcan con
@unittest.skipUnless y se omiten automáticamente si los archivos
no están presentes.

Universidad Santo Tomás - Maestría en Ciencia de Datos
Autor: James Mosquera Rentería
"""

import sys
import os
import unittest
from dataclasses import replace
from unittest.mock import MagicMock

import numpy as np

# Agregar directorio raíz del proyecto al path
RUTA_PROYECTO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RUTA_PROYECTO)

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
# Rutas de recursos
# ---------------------------------------------------------------------------
RUTA_MODELO = os.path.join(RUTA_PROYECTO, "modelo_desercion_final.pkl")
RUTA_SCALER = os.path.join(RUTA_PROYECTO, "scaler_desercion.pkl")
RUTA_DATOS = os.path.join(RUTA_PROYECTO, "data.csv")

MODELOS_DISPONIBLES = os.path.exists(RUTA_MODELO) and os.path.exists(RUTA_SCALER)
DATASET_DISPONIBLE = os.path.exists(RUTA_DATOS)

# Fixture base
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
# SECCIÓN 1: Carga de modelos reales  (I-01 – I-05)
# ===========================================================================
@unittest.skipUnless(MODELOS_DISPONIBLES, "Archivos .pkl no disponibles")
class TestIntegracionCargaModelo(unittest.TestCase):
    """Pruebas de integración: carga correcta de modelo y scaler reales."""

    @classmethod
    def setUpClass(cls):
        cls.modelo, cls.scaler = cargar_modelos(RUTA_MODELO, RUTA_SCALER)

    def test_I01_modelos_reales_cargan_sin_excepciones(self):
        """I-01: cargar_modelos no lanza excepciones con los .pkl reales."""
        self.assertIsNotNone(self.modelo)
        self.assertIsNotNone(self.scaler)

    def test_I02_modelo_tiene_metodo_predict(self):
        """I-02: El modelo cargado expone el método predict."""
        self.assertTrue(hasattr(self.modelo, "predict"))

    def test_I03_modelo_tiene_metodo_predict_proba(self):
        """I-03: El modelo cargado expone el método predict_proba."""
        self.assertTrue(hasattr(self.modelo, "predict_proba"))

    def test_I04_scaler_tiene_metodo_transform(self):
        """I-04: El scaler cargado expone el método transform."""
        self.assertTrue(hasattr(self.scaler, "transform"))

    def test_I05_scaler_acepta_vector_de_36_caracteristicas(self):
        """I-05: scaler.transform no falla con un vector de shape (1, 36)."""
        resultado = self.scaler.transform(np.zeros((1, 36)))
        self.assertEqual(resultado.shape, (1, 36))


# ===========================================================================
# SECCIÓN 2: Flujo completo con modelos reales  (I-06 – I-10)
# ===========================================================================
@unittest.skipUnless(MODELOS_DISPONIBLES, "Archivos .pkl no disponibles")
class TestIntegracionFlujoPredecirConModeloReal(unittest.TestCase):
    """Pruebas de integración: pipeline completo con modelo y scaler reales."""

    @classmethod
    def setUpClass(cls):
        cls.modelo, cls.scaler = cargar_modelos(RUTA_MODELO, RUTA_SCALER)

    def test_I06_flujo_completo_perfil_favorable(self):
        """I-06: Estudiante con perfil favorable produce un resultado coherente."""
        datos = DatosEstudiante(
            edad=20, genero=0, estado_civil=1, becario=1,
            deudor=0, matricula_dia=1, materias_aprobadas_s1=6,
            promedio_s1=15.0, nota_admision=160.0,
        )
        resultado = ejecutar_prediccion_completa(datos, self.modelo, self.scaler)
        self.assertIsInstance(resultado, ResultadoPrediccion)
        self.assertIn(resultado.es_alto_riesgo, [True, False])
        self.assertGreaterEqual(resultado.probabilidad_desercion, 0.0)
        self.assertLessEqual(resultado.probabilidad_desercion, 1.0)

    def test_I07_flujo_completo_perfil_desfavorable(self):
        """I-07: Estudiante con perfil desfavorable produce un resultado coherente."""
        datos = DatosEstudiante(
            edad=35, genero=1, estado_civil=2, becario=0,
            deudor=1, matricula_dia=0, materias_aprobadas_s1=0,
            promedio_s1=0.0, nota_admision=100.0,
        )
        resultado = ejecutar_prediccion_completa(datos, self.modelo, self.scaler)
        self.assertIsInstance(resultado, ResultadoPrediccion)
        self.assertGreaterEqual(resultado.probabilidad_desercion, 0.0)
        self.assertLessEqual(resultado.probabilidad_desercion, 1.0)

    def test_I08_probabilidades_suman_1_con_modelo_real(self):
        """I-08: probabilidad_desercion + probabilidad_permanencia ~= 1.0."""
        resultado = ejecutar_prediccion_completa(DATOS_BASE, self.modelo, self.scaler)
        total = resultado.probabilidad_desercion + resultado.probabilidad_permanencia
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_I09_recomendaciones_coherentes_con_prediccion(self):
        """I-09: Las recomendaciones son coherentes con la etiqueta de riesgo."""
        resultado = ejecutar_prediccion_completa(DATOS_BASE, self.modelo, self.scaler)
        textos = " ".join(resultado.recomendaciones).lower()
        if resultado.es_alto_riesgo:
            self.assertTrue(
                "tutor" in textos or "mentoría" in textos or "seguimiento" in textos
            )
        else:
            self.assertIn("monitoreo", textos)

    def test_I10_prediccion_determinista_mismos_datos(self):
        """I-10: Dos predicciones con los mismos datos producen el mismo resultado."""
        r1 = ejecutar_prediccion_completa(DATOS_BASE, self.modelo, self.scaler)
        r2 = ejecutar_prediccion_completa(DATOS_BASE, self.modelo, self.scaler)
        self.assertEqual(r1.es_alto_riesgo, r2.es_alto_riesgo)
        self.assertAlmostEqual(
            r1.probabilidad_desercion, r2.probabilidad_desercion, places=10
        )


# ===========================================================================
# SECCIÓN 3: Coherencia con el dataset real  (I-11 – I-17)
# ===========================================================================
@unittest.skipUnless(DATASET_DISPONIBLE, "Dataset data.csv no disponible")
class TestIntegracionConDataset(unittest.TestCase):
    """Pruebas de integración: validación contra el dataset real."""

    @classmethod
    def setUpClass(cls):
        import pandas as pd
        cls.df = pd.read_csv(RUTA_DATOS, sep=";")
        if MODELOS_DISPONIBLES:
            cls.modelo, cls.scaler = cargar_modelos(RUTA_MODELO, RUTA_SCALER)
        else:
            cls.modelo = cls.scaler = None

    def test_I11_dataset_contiene_columnas_clave(self):
        """I-11: El CSV contiene todas las columnas que usa la aplicación."""
        columnas_requeridas = [
            "Marital status", "Gender", "Age at enrollment",
            "Admission grade", "Scholarship holder", "Debtor",
            "Tuition fees up to date",
            "Curricular units 1st sem (approved)",
            "Curricular units 1st sem (grade)",
            "Target",
        ]
        for col in columnas_requeridas:
            with self.subTest(columna=col):
                self.assertIn(col, self.df.columns)

    def test_I12_dataset_no_esta_vacio(self):
        """I-12: El dataset tiene al menos un registro."""
        self.assertGreater(len(self.df), 0)

    def test_I13_columna_target_presente(self):
        """I-13: La columna 'Target' existe en el dataset."""
        self.assertIn("Target", self.df.columns)

    def test_I14_target_solo_valores_esperados(self):
        """I-14: Los únicos valores en Target son Dropout, Graduate, Enrolled."""
        valores_validos = {"Dropout", "Graduate", "Enrolled"}
        valores_encontrados = set(self.df["Target"].unique())
        diferencia = valores_encontrados - valores_validos
        self.assertFalse(
            diferencia,
            f"Valores inesperados en Target: {diferencia}",
        )

    def test_I15_prediccion_exitosa_sobre_muestra_del_dataset(self):
        """I-15: Al menos una fila del dataset puede procesarse sin errores."""
        if self.modelo is None or self.scaler is None:
            self.skipTest("Modelos no disponibles")

        exitosas = 0
        for _, fila in self.df.head(20).iterrows():
            try:
                edad = max(17, min(70, int(fila["Age at enrollment"])))
                datos = DatosEstudiante(
                    edad=edad,
                    genero=int(fila["Gender"]),
                    estado_civil=int(fila["Marital status"]),
                    becario=int(fila["Scholarship holder"]),
                    deudor=int(fila["Debtor"]),
                    matricula_dia=int(fila["Tuition fees up to date"]),
                    materias_aprobadas_s1=int(
                        fila["Curricular units 1st sem (approved)"]
                    ),
                    promedio_s1=float(fila["Curricular units 1st sem (grade)"]),
                    nota_admision=float(fila["Admission grade"]),
                )
                ejecutar_prediccion_completa(datos, self.modelo, self.scaler)
                exitosas += 1
            except ValueError:
                pass  # Datos fuera de rango esperados en algunos registros

        self.assertGreater(exitosas, 0, "Ningún registro pudo predecirse")

    def test_I16_distribucion_clases_ninguna_supera_90_porciento(self):
        """I-16: Ninguna clase domina el dataset con más del 90% de registros."""
        conteos = self.df["Target"].value_counts(normalize=True)
        for clase, proporcion in conteos.items():
            with self.subTest(clase=clase):
                self.assertLess(
                    proporcion, 0.90,
                    f"Clase '{clase}' domina con {proporcion:.1%}",
                )

    def test_I17_mayoria_edades_en_rango_17_70(self):
        """I-17: Al menos el 95% de los registros tiene edad entre 17 y 70."""
        en_rango = self.df[
            (self.df["Age at enrollment"] >= 17)
            & (self.df["Age at enrollment"] <= 70)
        ]
        proporcion = len(en_rango) / len(self.df)
        self.assertGreater(
            proporcion, 0.95,
            f"Solo el {proporcion:.1%} de edades está en rango válido",
        )


# ===========================================================================
# SECCIÓN 4: Pipeline de extremo a extremo con mocks  (I-18 – I-23)
# ===========================================================================
class TestIntegracionPipelineConMocks(unittest.TestCase):
    """Pruebas de integración: pipeline completo usando mocks para los modelos."""

    def _mocks(self, clase=0, probs=(0.75, 0.25)):
        modelo = MagicMock()
        modelo.predict.return_value = np.array([clase])
        modelo.predict_proba.return_value = np.array([list(probs)])
        scaler = MagicMock()
        scaler.transform.return_value = np.zeros((1, 36))
        return modelo, scaler

    def test_I18_flujo_completo_con_mocks_bajo_riesgo(self):
        """I-18: Pipeline completo con mocks produce ResultadoPrediccion correcto."""
        modelo, scaler = self._mocks(clase=0, probs=(0.75, 0.25))
        resultado = ejecutar_prediccion_completa(DATOS_BASE, modelo, scaler)
        self.assertFalse(resultado.es_alto_riesgo)
        self.assertAlmostEqual(resultado.probabilidad_desercion, 0.25)
        self.assertAlmostEqual(resultado.probabilidad_permanencia, 0.75)
        self.assertIsInstance(resultado.recomendaciones, list)

    def test_I19_cada_paso_del_pipeline_produce_salida_correcta(self):
        """I-19: Cada etapa del pipeline (validar→vector→norm→predecir→rec) es coherente."""
        # 1. Validar
        es_valido, _ = validar_entrada(DATOS_BASE)
        self.assertTrue(es_valido)

        # 2. Construir vector
        vector = construir_vector_caracteristicas(DATOS_BASE)
        self.assertEqual(len(vector), 36)

        # 3. Normalizar (mock)
        scaler = MagicMock()
        scaler.transform.return_value = np.random.randn(1, 36)
        datos_norm = normalizar_datos(vector, scaler)
        self.assertEqual(datos_norm.shape, (1, 36))

        # 4. Predecir (mock)
        modelo = MagicMock()
        modelo.predict.return_value = np.array([1])
        modelo.predict_proba.return_value = np.array([[0.3, 0.7]])
        prediccion, probs = predecir_riesgo(datos_norm, modelo)

        # 5. Recomendaciones
        rec = obtener_recomendaciones(es_alto_riesgo=(prediccion == 1))

        self.assertEqual(prediccion, 1)
        self.assertAlmostEqual(float(sum(probs)), 1.0)
        self.assertGreater(len(rec), 0)

    def test_I20_estudiante_edad_minima_17(self):
        """I-20: Estudiante de 17 años se procesa sin errores."""
        modelo, scaler = self._mocks(clase=0, probs=(0.8, 0.2))
        datos = replace(DATOS_BASE, edad=17, promedio_s1=18.0, nota_admision=190.0)
        resultado = ejecutar_prediccion_completa(datos, modelo, scaler)
        self.assertIsNotNone(resultado)

    def test_I21_estudiante_edad_maxima_70(self):
        """I-21: Estudiante de 70 años se procesa sin errores."""
        modelo, scaler = self._mocks(clase=1, probs=(0.2, 0.8))
        datos = replace(DATOS_BASE, edad=70, deudor=1, matricula_dia=0,
                        materias_aprobadas_s1=0, promedio_s1=0.0)
        resultado = ejecutar_prediccion_completa(datos, modelo, scaler)
        self.assertTrue(resultado.es_alto_riesgo)

    def test_I22_multiples_estudiantes_consecutivos(self):
        """I-22: El sistema procesa múltiples estudiantes en secuencia sin estado residual."""
        estudiantes = [
            DatosEstudiante(20, 1, 1, 0, 0, 1, 5, 12.0, 130.0),
            DatosEstudiante(25, 0, 2, 1, 0, 1, 6, 14.0, 150.0),
            DatosEstudiante(30, 1, 1, 0, 1, 0, 2, 8.0, 110.0),
        ]
        clases = [0, 0, 1]
        resultados = []
        for datos, clase in zip(estudiantes, clases):
            modelo, scaler = self._mocks(
                clase=clase,
                probs=(0.8 - clase * 0.6, 0.2 + clase * 0.6),
            )
            resultados.append(
                ejecutar_prediccion_completa(datos, modelo, scaler)
            )

        self.assertEqual(len(resultados), 3)
        self.assertFalse(resultados[0].es_alto_riesgo)
        self.assertFalse(resultados[1].es_alto_riesgo)
        self.assertTrue(resultados[2].es_alto_riesgo)

    def test_I23_caso_frontera_becario_con_deudas(self):
        """I-23: Becario con deudas (caso contradictorio) se procesa sin error."""
        modelo, scaler = self._mocks(clase=1, probs=(0.4, 0.6))
        datos = DatosEstudiante(
            edad=22, genero=1, estado_civil=1,
            becario=1,   # becario...
            deudor=1,    # ...pero con deudas: caso frontera
            matricula_dia=0,
            materias_aprobadas_s1=3,
            promedio_s1=9.0,
            nota_admision=115.0,
        )
        resultado = ejecutar_prediccion_completa(datos, modelo, scaler)
        self.assertIsNotNone(resultado)
        self.assertTrue(resultado.es_alto_riesgo)


# ===========================================================================
# Punto de entrada
# ===========================================================================
if __name__ == "__main__":
    unittest.main(verbosity=2)
