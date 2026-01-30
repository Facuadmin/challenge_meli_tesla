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
│   ├── Predictive_Maintenance_Devices.ipynb  # Notebook principal
│   ├── utils_classifier.py             # Módulo con funciones de clasificación
│   ├── full_devices.csv                # Dataset de dispositivos
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

**Objetivo:** Predecir la probabilidad de falla de dispositivos en galpones Full de Mercado Libre para optimizar costos de mantenimiento.

**Notebook:** `3_PREVISION_FALLOS/Predictive_Maintenance_Devices.ipynb`

**Matriz de costos:**
| Escenario | Costo |
|-----------|-------|
| Falla no prevenida (FN) | 1.0 |
| Mantenimiento preventivo (TP, FP) | 0.5 |
| Sin costo (TN) | 0.0 |

**Modelos evaluados:**
- Logistic Regression (baseline)
- Random Forest
- XGBoost
- LightGBM
- CatBoost

**Técnicas de balanceo de clases:**
- SMOTE
- ADASYN
- SMOTE-ENN
- SMOTE-Tomek

**Optimización de hiperparámetros:** Optuna con TPE Sampler

**Métricas de evaluación:**
- ROC-AUC
- Precision / Recall / F1
- Cost-based evaluation
- Calibration curves

---

## Instalación

### 1. Crear entorno virtual (recomendado)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
.\venv\Scripts\activate   # Windows
```

### 2. Instalar dependencias

```bash
cd challenge
pip install -r requirements.txt
```

### 3. Ejecutar notebooks

```bash
jupyter notebook
# o
jupyter lab
```

---

## Requisitos del Sistema

- **Python:** 3.9+
- **RAM:** 8GB mínimo (16GB recomendado para modelos de embeddings)
- **Espacio en disco:** ~5GB (para modelos pre-entrenados)

---

## Uso Rápido

### Ejercicio 1 - EDA
```python
from utils_eda import OfertasEDA, PerformanceGeneral

eda = OfertasEDA('ofertas_relampago.csv')
success_rates = PerformanceGeneral.get_success_rates(eda.df)
PerformanceGeneral.plot_success_rates(eda.df)
```

### Ejercicio 2 - Similitud
```python
from utils_similarity import ProductSimilarity, generar_output_similitud

# Generar output de similitud
output = generar_output_similitud(df_test, modelo='sbert', top_k=1000)
output.to_csv('output_similitud.csv', index=False)

# O calcular similitud entre dos productos
calculator = ProductSimilarity('sbert')
score = calculator.get_similarity("Nike Air Max", "Zapatillas Nike")
```

### Ejercicio 3 - Predicción de Fallas
```python
import joblib
from utils_classifier import calculate_cost

# Cargar modelo entrenado
model = joblib.load('predictive_maintenance_model_optimized.pkl')

# Predecir
predictions = model.predict(X_test)
total_cost, breakdown = calculate_cost(y_test, predictions)
```

---

## Aspectos Evaluados

| Aspecto | Ejercicio(s) |
|---------|--------------|
| Capacidad analítica y exploración | 1, 2, 3 |
| Visualización de resultados | 1, 2, 3 |
| Feature engineering | 2, 3 |
| Modelado ML | 2, 3 |
| Análisis de performance | 2, 3 |
| Buenas prácticas de desarrollo | 1, 2, 3 |
| ML en producción | 3 |

---

## Autor

Challenge completado como parte del proceso de selección para el equipo de Data & Analytics de Mercado Libre.
