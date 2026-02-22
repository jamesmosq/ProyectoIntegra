# Sistema de Alerta Temprana de Deserción Estudiantil

Proyecto de la **Maestría en Ciencia de Datos** — Universidad Santo Tomás
Autor: **Ing. James Mosquera Rentería**

---

## Descripción

Sistema basado en Machine Learning que predice el riesgo de deserción de estudiantes universitarios a partir de variables académicas, socioeconómicas y demográficas. Permite a equipos de bienestar estudiantil identificar casos en riesgo y tomar acciones preventivas de forma temprana.

---

## Estructura del proyecto

```
ProyectoIntegra/
├── app_desercion.py          # Aplicación web Streamlit (interfaz de usuario)
├── predictor_logica.py       # Módulo de lógica de negocio (testeable)
├── ejecutar_pruebas.py       # Runner de pruebas unitarias e integradas
├── data.csv                  # Dataset de 4 424 estudiantes (separador: ;)
├── modelo_desercion_final.pkl # Modelo de clasificación entrenado
├── scaler_desercion.pkl      # StandardScaler entrenado
├── requirements.txt          # Dependencias Python
└── tests/
    ├── __init__.py
    ├── test_unitarias.py     # 57 pruebas unitarias (U-01 a U-57)
    └── test_integradas.py    # 23 pruebas de integración (I-01 a I-23)
```

---

## Requisitos

- Python 3.8 o superior

Instalar dependencias:

```bash
pip install -r requirements.txt
```

`requirements.txt` incluye: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `joblib`.

---

## Ejecución de la aplicación

```bash
streamlit run app_desercion.py
```

La aplicación abre en el navegador en `http://localhost:8501`. El formulario solicita:

| Campo | Rango |
|---|---|
| Edad al matricularse | 17 – 70 |
| Género | Femenino / Masculino |
| Estado civil | Soltero / Casado / Viudo / Divorciado / Unión libre / Separado |
| ¿Es becario? | Sí / No |
| ¿Tiene deudas? | Sí / No |
| ¿Matrícula al día? | Sí / No |
| Materias aprobadas (1er semestre) | 0 – 20 |
| Promedio (1er semestre) | 0.0 – 20.0 |
| Nota de admisión | 0.0 – 200.0 |

### Resultado

- **ALTO RIESGO**: probabilidad de deserción + recomendaciones de intervención (tutor, apoyo financiero, mentoría, seguimiento).
- **BAJO RIESGO**: probabilidad de permanencia + recomendaciones de seguimiento.

---

## Métricas del modelo

| Métrica | Valor aproximado |
|---|---|
| Recall | 70 – 80 % |
| Precision | 60 – 70 % |
| F1-Score | 65 – 75 % |

---

## Ejecución de pruebas

### Todas las pruebas (unitarias + integradas)

```bash
python ejecutar_pruebas.py
```

### Solo pruebas unitarias

```bash
python ejecutar_pruebas.py unitarias
```

### Solo pruebas de integración

```bash
python ejecutar_pruebas.py integradas
```

Resultado esperado:

```
Ran 80 tests in ~3s
OK
RESUMEN: 80 ejecutadas | 0 fallidas | 0 errores | 80 exitosas
```

---

## Casos de prueba

### Pruebas unitarias (`tests/test_unitarias.py`) — 57 casos

| ID | Clase | Descripción |
|---|---|---|
| U-01 | `TestValidarEntrada` | Datos válidos → devuelve `(True, '')` |
| U-02 | `TestValidarEntrada` | Edad mínima 17 es aceptada |
| U-03 | `TestValidarEntrada` | Edad máxima 70 es aceptada |
| U-04 | `TestValidarEntrada` | Edad 16 es rechazada |
| U-05 | `TestValidarEntrada` | Edad 71 es rechazada |
| U-06 | `TestValidarEntrada` | Género 0 (Femenino) es aceptado |
| U-07 | `TestValidarEntrada` | Género 1 (Masculino) es aceptado |
| U-08 | `TestValidarEntrada` | Género 2 (inválido) es rechazado |
| U-09 | `TestValidarEntrada` | Estado civil 1–6 son todos aceptados |
| U-10 | `TestValidarEntrada` | Estado civil 0 es rechazado |
| U-11 | `TestValidarEntrada` | Estado civil 7 es rechazado |
| U-12 | `TestValidarEntrada` | Promedio 0.0 es aceptado |
| U-13 | `TestValidarEntrada` | Promedio 20.0 es aceptado |
| U-14 | `TestValidarEntrada` | Promedio −0.1 es rechazado |
| U-15 | `TestValidarEntrada` | Promedio 20.1 es rechazado |
| U-16 | `TestValidarEntrada` | Nota de admisión 0.0 es aceptada |
| U-17 | `TestValidarEntrada` | Nota de admisión 200.0 es aceptada |
| U-18 | `TestValidarEntrada` | Nota de admisión 201 es rechazada |
| U-19 | `TestValidarEntrada` | 0 materias aprobadas es aceptado |
| U-20 | `TestValidarEntrada` | Materias aprobadas negativas son rechazadas |
| U-21 | `TestConstruirVector` | El vector tiene 36 elementos |
| U-22 | `TestConstruirVector` | El resultado es un `np.ndarray` |
| U-23 | `TestConstruirVector` | El dtype es `float64` |
| U-24 | `TestConstruirVector` | `estado_civil` en posición [0] |
| U-25 | `TestConstruirVector` | `nota_admision` en posición [12] |
| U-26 | `TestConstruirVector` | `deudor` en posición [15] |
| U-27 | `TestConstruirVector` | `matricula_dia` en posición [16] |
| U-28 | `TestConstruirVector` | `genero` en posición [17] |
| U-29 | `TestConstruirVector` | `becario` en posición [18] |
| U-30 | `TestConstruirVector` | `edad` en posición [19] |
| U-31 | `TestConstruirVector` | `materias_aprobadas_s1` en posición [24] |
| U-32 | `TestConstruirVector` | Posiciones no asignadas son 0.0 |
| U-33 | `TestNormalizarDatos` | Resultado tiene shape `(1, 36)` |
| U-34 | `TestNormalizarDatos` | `scaler.transform` se llama una vez |
| U-35 | `TestNormalizarDatos` | El vector correcto se pasa al scaler |
| U-36 | `TestPredecirRiesgo` | Predicción 1 → alto riesgo |
| U-37 | `TestPredecirRiesgo` | Predicción 0 → bajo riesgo |
| U-38 | `TestPredecirRiesgo` | Las probabilidades tienen 2 elementos |
| U-39 | `TestPredecirRiesgo` | Las probabilidades suman 1.0 |
| U-40 | `TestPredecirRiesgo` | La predicción es de tipo `int` |
| U-41 | `TestObtenerRecomendaciones` | Alto riesgo devuelve lista no vacía |
| U-42 | `TestObtenerRecomendaciones` | Bajo riesgo devuelve lista no vacía |
| U-43 | `TestObtenerRecomendaciones` | Alto riesgo menciona tutor |
| U-44 | `TestObtenerRecomendaciones` | Alto riesgo menciona seguimiento |
| U-45 | `TestObtenerRecomendaciones` | Bajo riesgo menciona monitoreo |
| U-46 | `TestObtenerRecomendaciones` | El retorno es de tipo `list` |
| U-47 | `TestObtenerRecomendaciones` | Cada elemento es un `str` |
| U-48 | `TestObtenerRecomendaciones` | Alto riesgo ≥ recomendaciones que bajo riesgo |
| U-49 | `TestCargarModelos` | Carga exitosa devuelve modelo y scaler |
| U-50 | `TestCargarModelos` | `FileNotFoundError` se propaga al caller |
| U-51 | `TestCargarModelos` | `joblib.load` se llama exactamente dos veces |
| U-52 | `TestEjecutarPrediccionCompleta` | Retorno es de tipo `ResultadoPrediccion` |
| U-53 | `TestEjecutarPrediccionCompleta` | Predicción 1 → `es_alto_riesgo=True` |
| U-54 | `TestEjecutarPrediccionCompleta` | Predicción 0 → `es_alto_riesgo=False` |
| U-55 | `TestEjecutarPrediccionCompleta` | Probabilidades asignadas correctamente |
| U-56 | `TestEjecutarPrediccionCompleta` | Datos inválidos lanzan `ValueError` |
| U-57 | `TestEjecutarPrediccionCompleta` | Resultado incluye recomendaciones no vacías |

### Pruebas de integración (`tests/test_integradas.py`) — 23 casos

| ID | Clase | Descripción |
|---|---|---|
| I-01 | `TestIntegracionCargaModelo` | Los archivos `.pkl` cargan sin excepciones |
| I-02 | `TestIntegracionCargaModelo` | El modelo expone el método `predict` |
| I-03 | `TestIntegracionCargaModelo` | El modelo expone el método `predict_proba` |
| I-04 | `TestIntegracionCargaModelo` | El scaler expone el método `transform` |
| I-05 | `TestIntegracionCargaModelo` | El scaler acepta vectores de 36 características |
| I-06 | `TestFlujoPredecirModeloReal` | Perfil favorable produce resultado coherente |
| I-07 | `TestFlujoPredecirModeloReal` | Perfil desfavorable produce resultado coherente |
| I-08 | `TestFlujoPredecirModeloReal` | Probabilidades suman 1.0 con modelo real |
| I-09 | `TestFlujoPredecirModeloReal` | Recomendaciones coherentes con la predicción |
| I-10 | `TestFlujoPredecirModeloReal` | Predicción determinista con los mismos datos |
| I-11 | `TestIntegracionConDataset` | El CSV contiene todas las columnas requeridas |
| I-12 | `TestIntegracionConDataset` | El dataset no está vacío |
| I-13 | `TestIntegracionConDataset` | La columna `Target` existe |
| I-14 | `TestIntegracionConDataset` | `Target` solo tiene valores `Dropout/Graduate/Enrolled` |
| I-15 | `TestIntegracionConDataset` | Al menos una fila del dataset se puede predecir |
| I-16 | `TestIntegracionConDataset` | Ninguna clase supera el 90 % del dataset |
| I-17 | `TestIntegracionConDataset` | El 95 % de las edades está entre 17 y 70 |
| I-18 | `TestIntegracionPipelineConMocks` | Pipeline completo con mocks produce resultado correcto |
| I-19 | `TestIntegracionPipelineConMocks` | Cada etapa del pipeline produce salida coherente |
| I-20 | `TestIntegracionPipelineConMocks` | Estudiante con edad mínima (17) se procesa sin error |
| I-21 | `TestIntegracionPipelineConMocks` | Estudiante con edad máxima (70) se procesa sin error |
| I-22 | `TestIntegracionPipelineConMocks` | Múltiples estudiantes consecutivos sin estado residual |
| I-23 | `TestIntegracionPipelineConMocks` | Caso frontera: becario con deudas se procesa sin error |

---

## Arquitectura

```
app_desercion.py  (UI Streamlit)
       │
       ▼
predictor_logica.py  (Lógica de negocio)
  ├── DatosEstudiante          (dataclass de entrada)
  ├── ResultadoPrediccion      (dataclass de salida)
  ├── validar_entrada()
  ├── construir_vector_caracteristicas()
  ├── normalizar_datos()
  ├── predecir_riesgo()
  ├── obtener_recomendaciones()
  ├── cargar_modelos()
  └── ejecutar_prediccion_completa()  ← pipeline completo
```

---

## Dataset

- **Archivo:** `data.csv` (separador `;`)
- **Registros:** 4 424 estudiantes
- **Características:** 36 variables académicas, demográficas y socioeconómicas
- **Variable objetivo:** `Target` → `Dropout` / `Graduate` / `Enrolled`
- **Fuente:** Conjunto de datos público de deserción universitaria (Portugal)
