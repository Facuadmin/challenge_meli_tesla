# Data Science Challenge
**Data & Analytics Team - Mercado Libre**

Este repositorio contiene la solución a los 3 ejercicios del desafío técnico de Data Science.

---

## Estructura del Proyecto

```
challenge/
├── 1_EDA/                              # Ejercicio 1: Análisis Exploratorio
│   ├── EDA_Ofertas_Relampago.ipynb     # Notebook principal del EDA
│   ├── utils_eda.py                    # Módulo con funciones de análisis
│   └── ofertas_relampago.csv           # Dataset de ofertas
│
├── 2_SIMILITUD/                        # Ejercicio 2: Similitud entre productos
│   ├── Similitud_Entrega_v2.ipynb      # Notebook principal (solución final)
│   ├── SBERT_vs_E5_Comparison_v2.ipynb # Comparación de modelos SBERT vs E5
│   ├── utils_similarity.py             # Módulo con clases de similitud
│   ├── items_titles.csv                # Dataset de entrenamiento (30K productos)
│   ├── items_titles_test.csv           # Dataset de test (10K productos)
│   └── output_similitud.csv            # Output con pares similares
│
├── 3_PREVISION_FALLOS/                 # Ejercicio 3: Predicción de fallas
│   ├── EDA_Predictive_Maintenance.ipynb      # Análisis exploratorio de datos
│   ├── Modelo_Final.ipynb                    # Notebook principal del modelo
│   ├── utils_classifier.py                   # Módulo con funciones de clasificación
│   ├── full_devices.csv                      # Dataset de dispositivos
│   └── predictive_maintenance_model_optimized.pkl  # Modelo entrenado
│
├── requirements.txt                    # Dependencias del proyecto
└── README.md                           # Este archivo
```

---

## Ejercicios

### 1. Explorar las Ofertas Relámpago - EDA

**Objetivo:** Realizar un análisis exploratorio sobre ofertas relámpago de Mercado Libre, generando insights accionables.

**Notebook:** `1_EDA/EDA_Ofertas_Relampago.ipynb`

**Temáticas analizadas:**
- **Performance y Resultados:** Tasas de éxito, análisis de ofertas "zombie"
- **Análisis Temporal:** Patrones por hora, día de semana, duración óptima
- **Categorías y Dominios:** Verticales, dominios problemáticos, análisis Pareto
- **Pricing, GMV y Velocidad:** Ticket promedio, GMV/hora, top performers
- **Stock y Operaciones:** Stock óptimo, eficiencia, sobreventas
- **Estrategia e Impacto:** Free shipping, riesgo operativo, FOMO

**Métricas clave definidas:**
| Término | Definición |
|---------|------------|
| Conversión | % de ofertas con al menos 1 venta |
| Zombie | Oferta sin ventas |
| Sellout | Oferta que agotó 100% del stock |
| Oversell | Oferta con ventas > stock comprometido |
| Sell-Through Rate | % del stock vendido |
| GMV | Gross Merchandise Value |

---

### 2. Similitud entre Productos

**Objetivo:** Generar pares de productos similares basándose en sus títulos, utilizando técnicas de NLP y embeddings.

**Notebooks:**
- `2_SIMILITUD/Similitud_Entrega_v2.ipynb` - Solución principal
- `2_SIMILITUD/SBERT_vs_E5_Comparison_v2.ipynb` - Comparación de modelos

**Modelos implementados:**
| Modelo | Arquitectura | Dimensión | Características |
|--------|--------------|-----------|-----------------|
| SBERT | paraphrase-multilingual-mpnet-base-v2 | 768 | Optimizado para paráfrasis |
| E5 | intfloat/multilingual-e5-base | 768 | Embeddings universales |
| Word2Vec | Entrenado en corpus | 100 | Skip-gram, promedio de palabras |
| FastText | Entrenado en corpus | 100 | N-grams de caracteres |

**Output esperado:**
```
| ITE_ITEM_TITLE | ITE_ITEM_TITLE_2 | Score Similitud (0,1) |
|----------------|------------------|----------------------|
| Producto A     | Producto B       | 0.9543               |
```

**Funcionalidades del módulo `utils_similarity.py`:**
- Preprocesamiento de títulos (normalización, limpieza)
- Clases para cada modelo: `ProductSimilarity`, `Word2VecSimilarity`, `FastTextSimilarity`
- Reducción de dimensionalidad (PCA, t-SNE)
- Visualizaciones 3D interactivas con Plotly
- Clustering con K-Means
- Comparación entre modelos (`ModelComparator`)

---

### 3. Previsión de Fallas - Mantenimiento Predictivo

**Objetivo:** Predecir la probabilidad de falla de dispositivos IoT en galpones Full de Mercado Libre para minimizar costos operativos.

**Notebooks:**
- `3_PREVISION_FALLOS/EDA_Predictive_Maintenance.ipynb` - Análisis exploratorio
- `3_PREVISION_FALLOS/Modelo_Final.ipynb` - Modelo principal

**Matriz de costos:**
| Escenario | Costo | Descripción |
|-----------|-------|-------------|
| Falla no detectada (FN) | 1.0 | El dispositivo falla sin mantenimiento preventivo |
| Mantenimiento preventivo (TP/FP) | 0.5 | Se realiza mantenimiento (correcto o innecesario) |
| Sin costo (TN) | 0.0 | No hay falla y no se hace mantenimiento |

**Implicación clave:** Los FN cuestan el doble que los FP → Priorizar Recall sobre Precision

**Modelo Final (V7):**
| Parámetro | Valor |
|-----------|-------|
| Modelo | BalancedBaggingClassifier |
| n_estimators | 93 |
| Threshold | 0.85 |
| Ventana de detección | 30 días |

**Feature Engineering:**
- Features temporales (día de semana, mes, semana del año)
- Rolling statistics (media móvil, máximo) con ventanas de 3, 7 y 14 días
- Z-scores por dispositivo
- Atributos clave identificados: `attribute2`, `attribute4`, `attribute7`

**Técnicas aplicadas:**
- Split temporal (respeta orden cronológico para evitar data leakage)
- BalancedBaggingClassifier para manejo de clases desbalanceadas
- Optimización de hiperparámetros con Optuna (TPE Sampler)
- Análisis de threshold para optimizar costo operativo

**Resultados:**
- Recall: 73.9% (detecta 17 de 23 fallas en test)
- Con threshold óptimo: Ahorro del 95.7% vs baseline

**Funcionalidades del módulo `utils_classifier.py`:**
- Carga y preprocesamiento de datos de telemetría
- Feature engineering temporal y rolling statistics
- Entrenamiento y evaluación con ventana de detección
- Visualizaciones: distribución de probabilidades, evolución temporal, matriz de confusión
- Análisis de threshold y comparación de costos
- Persistencia del modelo

---

## Autor
Facundo Maldonado