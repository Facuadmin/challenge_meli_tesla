"""
Utilidades para cálculo de similitud entre productos.

Este módulo contiene funciones y clases reutilizables para:
- Preprocesamiento de títulos de productos
- Generación de embeddings con modelos SBERT y E5
- Cálculo de similitud entre productos
- Reducción de dimensionalidad y visualización 3D
- Clustering de productos
- Comparación de modelos
"""

import re
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Union
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# PREPROCESAMIENTO
# =============================================================================

def preprocess_title(text: str) -> str:
    """
    Preprocesa el título del producto para normalización.
    
    Args:
        text: Título original del producto
    
    Returns:
        Título preprocesado (lowercase, sin caracteres especiales)
    
    Example:
        >>> preprocess_title("Tênis Nike Air Max - Masculino!")
        "tênis nike air max masculino"
    """
    if pd.isna(text):
        return ""
    text = text.lower()
    # Mantener caracteres alfanuméricos portugueses/españoles
    text = re.sub(r'[^a-záàâãéèêíïóôõúüç\s0-9]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_dataframe(
    df: pd.DataFrame,
    column: str = 'ITE_ITEM_TITLE',
    output_column: str = 'title_clean'
) -> pd.DataFrame:
    """
    Aplica preprocesamiento a una columna del DataFrame.
    
    Args:
        df: DataFrame con los datos
        column: Nombre de la columna a preprocesar
        output_column: Nombre de la columna de salida
    
    Returns:
        DataFrame con columna preprocesada añadida
    """
    df = df.copy()
    df[output_column] = df[column].apply(preprocess_title)
    return df


# =============================================================================
# REDUCCIÓN DE DIMENSIONALIDAD
# =============================================================================

def reduce_dimensions_3d(
    embeddings: np.ndarray,
    method: str = 'pca',
    perplexity: int = 30,
    max_iter: int = 1000,
    pca_components_for_tsne: int = 50,
    random_state: int = 42
) -> np.ndarray:
    """
    Reduce embeddings a 3 dimensiones para visualización.
    
    Args:
        embeddings: Matriz de embeddings (n_samples, n_features)
        method: 'pca' o 'tsne'
        perplexity: Perplexity para t-SNE
        max_iter: Máximo de iteraciones para t-SNE
        pca_components_for_tsne: Componentes PCA intermedios para t-SNE
        random_state: Semilla para reproducibilidad
    
    Returns:
        Embeddings reducidos a 3D (n_samples, 3)
    
    Example:
        >>> embeddings_3d = reduce_dimensions_3d(embeddings, method='pca')
    """
    if method == 'pca':
        reducer = PCA(n_components=3, random_state=random_state)
        embeddings_3d = reducer.fit_transform(embeddings)
        variance_explained = reducer.explained_variance_ratio_.sum()
        print(f"Varianza explicada por PCA: {variance_explained:.2%}")
        
    elif method == 'tsne':
        # Primero reducir con PCA para acelerar t-SNE
        n_components = min(pca_components_for_tsne, embeddings.shape[1])
        pca = PCA(n_components=n_components, random_state=random_state)
        embeddings_pca = pca.fit_transform(embeddings)
        
        tsne = TSNE(
            n_components=3,
            random_state=random_state,
            perplexity=perplexity,
            max_iter=max_iter
        )
        embeddings_3d = tsne.fit_transform(embeddings_pca)
    else:
        raise ValueError(f"Método no soportado: {method}. Use 'pca' o 'tsne'")
    
    return embeddings_3d


# =============================================================================
# VISUALIZACIÓN
# =============================================================================

def plot_3d_clusters(
    embeddings_3d: np.ndarray,
    clusters: np.ndarray,
    titles: List[str],
    title_plot: str = 'Clusters de Productos',
    width: int = 900,
    height: int = 700,
    save_html: Optional[str] = None,
    auto_open: bool = True
):
    """
    Visualiza clusters en 3D.
    
    Args:
        embeddings_3d: Embeddings reducidos a 3D
        clusters: Array con etiquetas de cluster
        titles: Lista de títulos
        title_plot: Título del gráfico
    
    Returns:
        Figura de Plotly
    """
    import plotly.express as px
    
    df_plot = pd.DataFrame({
        'x': embeddings_3d[:, 0],
        'y': embeddings_3d[:, 1],
        'z': embeddings_3d[:, 2],
        'cluster': clusters.astype(str),
        'title': titles
    })
    
    fig = px.scatter_3d(
        df_plot,
        x='x', y='y', z='z',
        color='cluster',
        hover_data={'title': True, 'x': False, 'y': False, 'z': False},
        title=title_plot,
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    
    fig.update_traces(marker=dict(size=4, opacity=0.7))
    fig.update_layout(width=width, height=height)
    
    if save_html:
        # Nota: antes se guardaba a HTML; ya no es necesario.
        warnings.warn(
            "plot_3d_clusters(): 'save_html' está deprecado y se ignora (ya no se guardan HTML).",
            category=DeprecationWarning,
            stacklevel=2,
        )
    
    return fig


# =============================================================================
# CLUSTERING
# =============================================================================

def cluster_embeddings(
    embeddings: np.ndarray,
    n_clusters: int = 5,
    random_state: int = 42
) -> np.ndarray:
    """
    Aplica K-Means clustering a los embeddings.
    
    Args:
        embeddings: Matriz de embeddings
        n_clusters: Número de clusters
        random_state: Semilla para reproducibilidad
    
    Returns:
        Array con etiquetas de cluster
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(embeddings)
    print(f"Clustering con {n_clusters} clusters completado")
    return clusters


# =============================================================================
# CLASE BASE DE SIMILITUD
# =============================================================================

class BaseSimilarity:
    """Clase base para calculadores de similitud."""
    
    def __init__(self):
        self.embeddings = None
        self.titles = None
        self.titles_original = None
        self.model = None
    
    def fit(self, titles: List[str], titles_original: Optional[List[str]] = None):
        """Genera embeddings para una lista de títulos."""
        raise NotImplementedError
    
    def get_similarity(self, title1: str, title2: str) -> float:
        """Calcula la similitud entre dos títulos."""
        raise NotImplementedError
    
    def get_similarity_by_index(self, idx1: int, idx2: int) -> float:
        """
        Calcula similitud entre dos productos por índice.
        
        Args:
            idx1: Índice del primer producto
            idx2: Índice del segundo producto
        
        Returns:
            Score de similitud entre 0 y 1
        """
        if self.embeddings is None:
            raise ValueError("Debe ejecutar fit() primero para generar embeddings")
        
        similarity = cosine_similarity(
            [self.embeddings[idx1]],
            [self.embeddings[idx2]]
        )[0][0]
        
        return float(similarity)
    
    def get_top_similar_pairs(self, top_k: int = 1000) -> pd.DataFrame:
        """
        Obtiene los top-k pares más similares del dataset.
        
        Args:
            top_k: Número de pares a retornar
        
        Returns:
            DataFrame con columnas: ITE_ITEM_TITLE, ITE_ITEM_TITLE_2, Score Similitud (0,1)
        """
        if self.embeddings is None:
            raise ValueError("Debe ejecutar fit() primero")
        
        print(f"Calculando matriz de similitud ({len(self.titles):,} x {len(self.titles):,})...")
        sim_matrix = cosine_similarity(self.embeddings)
        
        # Obtener índices del triángulo superior (sin diagonal)
        n = len(self.titles)
        upper_tri_indices = np.triu_indices(n, k=1)
        scores = sim_matrix[upper_tri_indices]
        
        # Ordenar y obtener top-k
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Crear DataFrame de resultados
        results = []
        for idx in top_indices:
            i = upper_tri_indices[0][idx]
            j = upper_tri_indices[1][idx]
            results.append({
                'ITE_ITEM_TITLE': self.titles_original[i],
                'ITE_ITEM_TITLE_2': self.titles_original[j],
                'Score Similitud (0,1)': round(scores[idx], 4)
            })
        
        return pd.DataFrame(results)
    
    def compare_products(self, product1: str, product2: str) -> dict:
        """
        Compara dos productos y retorna información detallada.
        
        Args:
            product1: Título del primer producto
            product2: Título del segundo producto
        
        Returns:
            Diccionario con información de similitud
        """
        similarity = self.get_similarity(product1, product2)
        
        return {
            'ITE_ITEM_TITLE': product1,
            'ITE_ITEM_TITLE_2': product2,
            'Score Similitud (0,1)': round(similarity, 4),
            'Modelo': self.__class__.__name__
        }
    
    def get_similarity_matrix(self) -> np.ndarray:
        """
        Obtiene la matriz completa de similitud.
        
        Returns:
            Matriz de similitud (n x n)
        """
        if self.embeddings is None:
            raise ValueError("Debe ejecutar fit() primero")
        
        return cosine_similarity(self.embeddings)


# =============================================================================
# SIMILITUD CON SENTENCE TRANSFORMERS (SBERT, E5)
# =============================================================================

class ProductSimilarity(BaseSimilarity):
    """
    Clase para calcular similitud entre productos usando Sentence Transformers.
    
    Modelos soportados:
    - 'sbert': Sentence-BERT (paraphrase-multilingual-MiniLM-L12-v2)
    - 'e5': E5 Multilingual (intfloat/multilingual-e5-base)
    - Custom: Cualquier modelo de sentence-transformers
    
    Example:
        >>> calculator = ProductSimilarity(model_name='sbert')
        >>> calculator.load_model()
        >>> calculator.fit(titles_clean, titles_original)
        >>> score = calculator.get_similarity("Produto A", "Produto B")
    """
    
    MODELS = {
        'sbert': 'paraphrase-multilingual-mpnet-base-v2',  # 768 dim
        'sbert-mini': 'paraphrase-multilingual-MiniLM-L12-v2',  # 384 dim (más rápido)
        'e5': 'intfloat/multilingual-e5-base',
        'e5-small': 'intfloat/multilingual-e5-small',
        'e5-large': 'intfloat/multilingual-e5-large',
    }
    
    def __init__(self, model_name: str = 'sbert'):
        """
        Inicializa el calculador de similitud.
        
        Args:
            model_name: Nombre del modelo ('sbert', 'e5', o path completo)
        """
        super().__init__()
        self.model_name = model_name
        self.model_path = self.MODELS.get(model_name, model_name)
    
    def load_model(self, verbose: bool = True):
        """Carga el modelo de embeddings.
        
        Args:
            verbose: Si es True, imprime mensajes de carga
        """
        from sentence_transformers import SentenceTransformer
        
        if verbose:
            print(f"Cargando modelo: {self.model_path}...")
        self.model = SentenceTransformer(self.model_path)
        if verbose:
            print(f"Modelo cargado. Dimensión: {self.model.get_sentence_embedding_dimension()}")
        return self
    
    def _prepare_text(self, text: str) -> str:
        """Prepara el texto según el modelo (agrega prefijo para E5)."""
        if 'e5' in self.model_name.lower():
            return f'passage: {text}'
        return text
    
    def _prepare_texts(self, texts: List[str]) -> List[str]:
        """Prepara lista de textos según el modelo."""
        return [self._prepare_text(t) for t in texts]
    
    def fit(
        self,
        titles: List[str],
        titles_original: Optional[List[str]] = None,
        batch_size: int = 64,
        show_progress: bool = True
    ):
        """
        Genera embeddings para una lista de títulos.
        
        Args:
            titles: Lista de títulos preprocesados
            titles_original: Lista de títulos originales (para output)
            batch_size: Tamaño de batch para encoding
            show_progress: Mostrar barra de progreso
        """
        if self.model is None:
            self.load_model()
        
        self.titles = titles
        self.titles_original = titles_original if titles_original else titles
        
        titles_to_encode = self._prepare_texts(titles)
        
        print(f"Generando embeddings para {len(titles):,} productos...")
        self.embeddings = self.model.encode(
            titles_to_encode,
            show_progress_bar=show_progress,
            batch_size=batch_size
        )
        print(f"Embeddings generados. Shape: {self.embeddings.shape}")
        return self
    
    def get_similarity(self, title1: str, title2: str) -> float:
        """
        Calcula la similitud entre dos títulos específicos.
        
        Args:
            title1: Primer título
            title2: Segundo título
        
        Returns:
            Score de similitud entre 0 y 1
        """
        if self.model is None:
            self.load_model(verbose=False)
        
        texts = self._prepare_texts([title1, title2])
        embeddings = self.model.encode(texts)
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        return float(similarity)
    
    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """
        Genera embeddings para una lista de textos.
        
        Args:
            texts: Lista de textos
            **kwargs: Argumentos para model.encode()
        
        Returns:
            Matriz de embeddings
        """
        if self.model is None:
            self.load_model()
        
        texts_prepared = self._prepare_texts(texts)
        return self.model.encode(texts_prepared, **kwargs)


# =============================================================================
# COMPARACIÓN DE MODELOS SBERT vs E5
# =============================================================================

class ModelComparator:
    """
    Clase para comparar embeddings de diferentes modelos (SBERT, E5, E5-finetuned).
    
    Proporciona métricas, visualizaciones y análisis comparativos detallados.
    
    Example:
        >>> comparator = ModelComparator()
        >>> comparator.add_model('sbert', ProductSimilarity('sbert'))
        >>> comparator.add_model('e5', ProductSimilarity('e5'))
        >>> comparator.fit_all(titles_clean, titles_original)
        >>> results = comparator.compare_all()
    """
    
    def __init__(self):
        """Inicializa el comparador de modelos."""
        self.models = {}
        self.embeddings = {}
        self.titles = None
        self.titles_original = None
        self._fit_done = False
    
    def add_model(
        self, 
        name: str, 
        model: Union['ProductSimilarity', str],
        custom_model_path: Optional[str] = None
    ):
        """
        Agrega un modelo para comparación.
        
        Args:
            name: Nombre identificador del modelo
            model: ProductSimilarity object o nombre del modelo ('sbert', 'e5', etc.)
            custom_model_path: Path a modelo fine-tuneado (opcional)
        """
        if isinstance(model, str):
            if custom_model_path:
                model = ProductSimilarity(model_name=custom_model_path)
            else:
                model = ProductSimilarity(model_name=model)
        
        self.models[name] = model
        print(f"Modelo '{name}' agregado para comparación")
        return self
    
    def fit_all(
        self, 
        titles: List[str], 
        titles_original: Optional[List[str]] = None,
        batch_size: int = 64,
        show_progress: bool = True
    ):
        """
        Genera embeddings para todos los modelos registrados.
        
        Args:
            titles: Lista de títulos preprocesados
            titles_original: Lista de títulos originales
            batch_size: Tamaño de batch
            show_progress: Mostrar progreso
        """
        self.titles = titles
        self.titles_original = titles_original if titles_original else titles
        
        for name, model in self.models.items():
            print(f"\n{'='*50}")
            print(f"Procesando modelo: {name}")
            print('='*50)
            
            model.load_model()
            model.fit(
                titles=titles,
                titles_original=self.titles_original,
                batch_size=batch_size,
                show_progress=show_progress
            )
            self.embeddings[name] = model.embeddings
        
        self._fit_done = True
        print(f"\n✓ Todos los modelos procesados ({len(self.models)} modelos)")
        return self
    
    def get_embedding_stats(self) -> pd.DataFrame:
        """
        Obtiene estadísticas de los embeddings de cada modelo.
        
        Returns:
            DataFrame con estadísticas por modelo
        """
        if not self._fit_done:
            raise ValueError("Debe ejecutar fit_all() primero")
        
        stats = []
        for name, emb in self.embeddings.items():
            stats.append({
                'Modelo': name,
                'Dimensión': emb.shape[1],
                'Samples': emb.shape[0],
                'Norma Media': np.linalg.norm(emb, axis=1).mean(),
                'Norma Std': np.linalg.norm(emb, axis=1).std(),
                'Media Componentes': emb.mean(),
                'Std Componentes': emb.std()
            })
        
        return pd.DataFrame(stats)
    
    def compare_similarity_distributions(
        self, 
        sample_size: int = 1000,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Compara distribuciones de similitud entre modelos.
        
        Args:
            sample_size: Número de muestras para el cálculo
            random_state: Semilla para reproducibilidad
        
        Returns:
            DataFrame con estadísticas de similitud por modelo
        """
        if not self._fit_done:
            raise ValueError("Debe ejecutar fit_all() primero")
        
        results = []
        for name, emb in self.embeddings.items():
            stats = calcular_estadisticas_similitud(
                emb, 
                sample_size=sample_size, 
                random_state=random_state
            )
            stats['Modelo'] = name
            results.append(stats)
        
        return pd.DataFrame(results)
    
    def compare_pairwise(
        self, 
        pairs: List[Tuple[int, int]]
    ) -> pd.DataFrame:
        """
        Compara scores de similitud para pares específicos.
        
        Args:
            pairs: Lista de tuplas (idx1, idx2) con índices de productos
        
        Returns:
            DataFrame con scores por modelo para cada par
        """
        if not self._fit_done:
            raise ValueError("Debe ejecutar fit_all() primero")
        
        results = []
        for idx1, idx2 in pairs:
            row = {
                'Producto 1': self.titles_original[idx1][:50],
                'Producto 2': self.titles_original[idx2][:50],
                'idx1': idx1,
                'idx2': idx2
            }
            
            for name, model in self.models.items():
                score = model.get_similarity_by_index(idx1, idx2)
                row[f'{name}_score'] = round(score, 4)
            
            results.append(row)
        
        return pd.DataFrame(results)
    
    def get_correlation_matrix(self, sample_size: int = 500) -> pd.DataFrame:
        """
        Calcula correlación entre scores de similitud de diferentes modelos.
        
        Args:
            sample_size: Número de pares a muestrear
        
        Returns:
            Matriz de correlación entre modelos
        """
        if not self._fit_done:
            raise ValueError("Debe ejecutar fit_all() primero")
        
        np.random.seed(42)
        n = len(self.titles)
        
        # Generar pares aleatorios
        pairs_idx = np.random.choice(n, size=(sample_size, 2), replace=True)
        pairs_idx = [(int(i), int(j)) for i, j in pairs_idx if i != j][:sample_size]
        
        # Calcular scores para cada modelo
        scores = {name: [] for name in self.models.keys()}
        
        for idx1, idx2 in pairs_idx:
            for name, model in self.models.items():
                score = model.get_similarity_by_index(idx1, idx2)
                scores[name].append(score)
        
        # Crear DataFrame y calcular correlación
        df_scores = pd.DataFrame(scores)
        return df_scores.corr()
    
    def plot_comparison_3d(
        self,
        method: str = 'pca',
        sample_size: Optional[int] = 2000,
        random_state: int = 42,
        width: int = 1400,
        height: int = 600
    ):
        """
        Crea visualización 3D comparativa side-by-side.
        
        Args:
            method: 'pca' o 'tsne'
            sample_size: Número de muestras (None = todas)
            random_state: Semilla para reproducibilidad
            width: Ancho total del gráfico
            height: Alto del gráfico
        
        Returns:
            Figura de Plotly con subplots
        """
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        
        if not self._fit_done:
            raise ValueError("Debe ejecutar fit_all() primero")
        
        # Muestrear si es necesario
        np.random.seed(random_state)
        n = len(self.titles)
        if sample_size and n > sample_size:
            idx_sample = np.random.choice(n, sample_size, replace=False)
        else:
            idx_sample = np.arange(n)
        
        model_names = list(self.models.keys())
        n_models = len(model_names)
        
        # Crear subplots
        fig = make_subplots(
            rows=1, cols=n_models,
            specs=[[{'type': 'scatter3d'} for _ in range(n_models)]],
            subplot_titles=[f'{name.upper()} - {method.upper()}' for name in model_names],
            horizontal_spacing=0.02
        )
        
        # Reducir dimensiones y plotear cada modelo
        for col, name in enumerate(model_names, 1):
            emb = self.embeddings[name][idx_sample]
            titles_sample = [self.titles_original[i] for i in idx_sample]
            
            # Reducir a 3D
            emb_3d = reduce_dimensions_3d(emb, method=method, random_state=random_state)
            
            # Calcular longitud de título para colorear
            word_counts = [len(t.split()) for t in titles_sample]
            
            fig.add_trace(
                go.Scatter3d(
                    x=emb_3d[:, 0],
                    y=emb_3d[:, 1],
                    z=emb_3d[:, 2],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=word_counts,
                        colorscale='Viridis',
                        opacity=0.7,
                        colorbar=dict(
                            title='Palabras',
                            x=0.33 * col if col < n_models else 1.0,
                            len=0.8
                        ) if col == n_models else None
                    ),
                    text=[t[:60] for t in titles_sample],
                    hoverinfo='text',
                    name=name
                ),
                row=1, col=col
            )
        
        fig.update_layout(
            title=f'Comparación de Espacios Vectoriales: {" vs ".join(model_names)}',
            width=width,
            height=height,
            showlegend=False
        )
        
        return fig
    
    def plot_similarity_distributions(self, sample_size: int = 1000):
        """
        Visualiza distribuciones de similitud comparativas.
        
        Args:
            sample_size: Número de pares a muestrear
        
        Returns:
            Figura de matplotlib
        """
        import matplotlib.pyplot as plt
        
        if not self._fit_done:
            raise ValueError("Debe ejecutar fit_all() primero")
        
        fig, axes = plt.subplots(1, len(self.models), figsize=(6*len(self.models), 5))
        if len(self.models) == 1:
            axes = [axes]
        
        colors = ['steelblue', 'darkorange', 'forestgreen', 'crimson']
        
        for ax, (name, emb), color in zip(axes, self.embeddings.items(), colors):
            # Muestrear y calcular similitudes
            np.random.seed(42)
            n = len(emb)
            if n > sample_size:
                idx = np.random.choice(n, sample_size, replace=False)
                emb_sample = emb[idx]
            else:
                emb_sample = emb
            
            sim_matrix = cosine_similarity(emb_sample)
            mask = np.triu(np.ones_like(sim_matrix, dtype=bool), k=1)
            sim_values = sim_matrix[mask]
            
            ax.hist(sim_values, bins=50, color=color, alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(sim_values), color='red', linestyle='--', 
                      label=f'Media: {np.mean(sim_values):.3f}')
            ax.axvline(np.median(sim_values), color='green', linestyle=':', 
                      label=f'Mediana: {np.median(sim_values):.3f}')
            ax.set_xlabel('Similitud Coseno')
            ax.set_ylabel('Frecuencia')
            ax.set_title(f'{name.upper()}')
            ax.legend()
        
        plt.suptitle('Distribución de Similitudes por Modelo', fontsize=14)
        plt.tight_layout()
        return fig
    
    def plot_score_comparison_scatter(
        self, 
        sample_size: int = 500,
        model1: Optional[str] = None,
        model2: Optional[str] = None
    ):
        """
        Scatter plot comparando scores de dos modelos.
        
        Args:
            sample_size: Número de pares a comparar
            model1: Nombre del primer modelo (default: primero agregado)
            model2: Nombre del segundo modelo (default: segundo agregado)
        
        Returns:
            Figura de plotly
        """
        import plotly.graph_objects as go
        
        if not self._fit_done:
            raise ValueError("Debe ejecutar fit_all() primero")
        
        model_names = list(self.models.keys())
        if model1 is None:
            model1 = model_names[0]
        if model2 is None:
            model2 = model_names[1] if len(model_names) > 1 else model_names[0]
        
        # Generar pares aleatorios
        np.random.seed(42)
        n = len(self.titles)
        pairs = []
        for _ in range(sample_size):
            i, j = np.random.randint(0, n, 2)
            if i != j:
                pairs.append((i, j))
        
        # Calcular scores
        scores1 = []
        scores2 = []
        hover_texts = []
        
        for idx1, idx2 in pairs[:sample_size]:
            s1 = self.models[model1].get_similarity_by_index(idx1, idx2)
            s2 = self.models[model2].get_similarity_by_index(idx1, idx2)
            scores1.append(s1)
            scores2.append(s2)
            hover_texts.append(
                f"{self.titles_original[idx1][:40]}...<br>"
                f"vs<br>"
                f"{self.titles_original[idx2][:40]}..."
            )
        
        # Calcular correlación
        correlation = np.corrcoef(scores1, scores2)[0, 1]
        
        fig = go.Figure()
        
        # Scatter de scores
        fig.add_trace(go.Scatter(
            x=scores1,
            y=scores2,
            mode='markers',
            marker=dict(
                size=6,
                color=np.abs(np.array(scores1) - np.array(scores2)),
                colorscale='RdYlBu_r',
                colorbar=dict(title='|Δ Score|'),
                opacity=0.7
            ),
            text=hover_texts,
            hoverinfo='text+x+y',
            name='Pares'
        ))
        
        # Línea de referencia (diagonal)
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Referencia (x=y)'
        ))
        
        fig.update_layout(
            title=f'Comparación de Scores: {model1.upper()} vs {model2.upper()}<br>'
                  f'<sub>Correlación: {correlation:.4f} | n={len(scores1)} pares</sub>',
            xaxis_title=f'Score {model1.upper()}',
            yaxis_title=f'Score {model2.upper()}',
            width=800,
            height=700,
            showlegend=True
        )
        
        return fig
    
    def get_top_disagreements(
        self, 
        n_pairs: int = 500,
        top_k: int = 20
    ) -> pd.DataFrame:
        """
        Encuentra pares donde los modelos más discrepan.
        
        Args:
            n_pairs: Número de pares a evaluar
            top_k: Número de mayores discrepancias a retornar
        
        Returns:
            DataFrame con pares de mayor discrepancia
        """
        if not self._fit_done:
            raise ValueError("Debe ejecutar fit_all() primero")
        
        model_names = list(self.models.keys())
        if len(model_names) < 2:
            raise ValueError("Necesita al menos 2 modelos para comparar discrepancias")
        
        # Generar pares aleatorios
        np.random.seed(42)
        n = len(self.titles)
        pairs = []
        for _ in range(n_pairs * 2):  # Generar extras por si hay duplicados
            i, j = np.random.randint(0, n, 2)
            if i != j and (i, j) not in pairs and (j, i) not in pairs:
                pairs.append((i, j))
            if len(pairs) >= n_pairs:
                break
        
        # Calcular scores y discrepancias
        results = []
        for idx1, idx2 in pairs:
            scores = {}
            for name in model_names:
                scores[name] = self.models[name].get_similarity_by_index(idx1, idx2)
            
            # Calcular máxima discrepancia
            score_values = list(scores.values())
            max_diff = max(score_values) - min(score_values)
            
            results.append({
                'Producto 1': self.titles_original[idx1],
                'Producto 2': self.titles_original[idx2],
                **{f'Score_{name}': round(s, 4) for name, s in scores.items()},
                'Max_Discrepancia': round(max_diff, 4)
            })
        
        df = pd.DataFrame(results)
        return df.nlargest(top_k, 'Max_Discrepancia')
    
    def generate_comparison_report(self, sample_size: int = 1000) -> dict:
        """
        Genera un reporte completo de comparación.
        
        Args:
            sample_size: Tamaño de muestra para estadísticas
        
        Returns:
            Diccionario con métricas y análisis
        """
        if not self._fit_done:
            raise ValueError("Debe ejecutar fit_all() primero")
        
        report = {
            'modelos': list(self.models.keys()),
            'n_productos': len(self.titles),
            'embedding_stats': self.get_embedding_stats().to_dict('records'),
            'similarity_distributions': self.compare_similarity_distributions(
                sample_size=sample_size
            ).to_dict('records'),
            'correlation_matrix': self.get_correlation_matrix(
                sample_size=sample_size
            ).to_dict()
        }
        
        return report


def compare_nearest_neighbors(
    model1: 'ProductSimilarity',
    model2: 'ProductSimilarity',
    query_indices: List[int],
    k: int = 5,
    titles_original: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compara los k vecinos más cercanos entre dos modelos.
    
    Args:
        model1: Primer modelo
        model2: Segundo modelo
        query_indices: Índices de productos a consultar
        k: Número de vecinos
        titles_original: Títulos originales
    
    Returns:
        DataFrame comparativo
    """
    if model1.embeddings is None or model2.embeddings is None:
        raise ValueError("Ambos modelos deben tener embeddings generados")
    
    titles = titles_original if titles_original else model1.titles_original
    
    results = []
    for idx in query_indices:
        # Calcular similitudes para model1
        sims1 = cosine_similarity([model1.embeddings[idx]], model1.embeddings)[0]
        top_k_1 = np.argsort(sims1)[::-1][1:k+1]  # Excluir el mismo
        
        # Calcular similitudes para model2
        sims2 = cosine_similarity([model2.embeddings[idx]], model2.embeddings)[0]
        top_k_2 = np.argsort(sims2)[::-1][1:k+1]
        
        # Calcular overlap
        overlap = len(set(top_k_1) & set(top_k_2))
        
        results.append({
            'Query': titles[idx][:50],
            'Query_idx': idx,
            'Model1_Top_K': [titles[i][:40] for i in top_k_1],
            'Model2_Top_K': [titles[i][:40] for i in top_k_2],
            f'Overlap@{k}': overlap,
            f'Overlap@{k}_%': round(overlap / k * 100, 1)
        })
    
    return pd.DataFrame(results)


def evaluate_ranking_consistency(
    model1: 'ProductSimilarity',
    model2: 'ProductSimilarity',
    n_queries: int = 100,
    k: int = 10
) -> dict:
    """
    Evalúa consistencia de rankings entre dos modelos.
    
    Args:
        model1: Primer modelo
        model2: Segundo modelo
        n_queries: Número de queries aleatorias
        k: Número de vecinos a comparar
    
    Returns:
        Diccionario con métricas de consistencia
    """
    np.random.seed(42)
    n = len(model1.embeddings)
    query_indices = np.random.choice(n, n_queries, replace=False)
    
    overlaps = []
    rank_correlations = []
    
    for idx in query_indices:
        # Similitudes model1
        sims1 = cosine_similarity([model1.embeddings[idx]], model1.embeddings)[0]
        ranks1 = np.argsort(sims1)[::-1]
        
        # Similitudes model2
        sims2 = cosine_similarity([model2.embeddings[idx]], model2.embeddings)[0]
        ranks2 = np.argsort(sims2)[::-1]
        
        # Top-k overlap
        top_k_1 = set(ranks1[1:k+1])
        top_k_2 = set(ranks2[1:k+1])
        overlap = len(top_k_1 & top_k_2) / k
        overlaps.append(overlap)
        
        # Rank correlation (Spearman) para top 100
        from scipy.stats import spearmanr
        top_100 = ranks1[1:101]
        ranks_in_model2 = [np.where(ranks2 == i)[0][0] for i in top_100]
        corr, _ = spearmanr(range(100), ranks_in_model2)
        rank_correlations.append(corr)
    
    return {
        f'mean_overlap@{k}': np.mean(overlaps),
        f'std_overlap@{k}': np.std(overlaps),
        'mean_rank_correlation': np.mean(rank_correlations),
        'std_rank_correlation': np.std(rank_correlations),
        'n_queries': n_queries
    }


# =============================================================================
# ANÁLISIS Y MÉTRICAS
# =============================================================================

def calcular_estadisticas_similitud(
    embeddings: np.ndarray,
    sample_size: int = 500,
    random_state: int = 42
) -> dict:
    """
    Calcula estadísticas de distribución de similitudes.
    
    Args:
        embeddings: Matriz de embeddings
        sample_size: Tamaño de muestra para cálculo
        random_state: Semilla para reproducibilidad
    
    Returns:
        Diccionario con estadísticas
    """
    np.random.seed(random_state)
    
    # Tomar muestra si es necesario
    if len(embeddings) > sample_size:
        idx_sample = np.random.choice(len(embeddings), sample_size, replace=False)
        embeddings_sample = embeddings[idx_sample]
    else:
        embeddings_sample = embeddings
    
    # Calcular matriz de similitud
    sim_matrix = cosine_similarity(embeddings_sample)
    
    # Obtener valores del triángulo superior (sin diagonal)
    mask = np.triu(np.ones_like(sim_matrix, dtype=bool), k=1)
    sim_values = sim_matrix[mask]
    
    return {
        'n_pairs': len(sim_values),
        'mean': float(np.mean(sim_values)),
        'std': float(np.std(sim_values)),
        'min': float(np.min(sim_values)),
        'max': float(np.max(sim_values)),
        'median': float(np.median(sim_values)),
        'percentile_25': float(np.percentile(sim_values, 25)),
        'percentile_75': float(np.percentile(sim_values, 75)),
    }


def analizar_longitud_titulos(df: pd.DataFrame, column: str = 'ITE_ITEM_TITLE') -> dict:
    """
    Analiza la longitud de los títulos.
    
    Args:
        df: DataFrame con los datos
        column: Nombre de la columna de títulos
    
    Returns:
        Diccionario con estadísticas
    """
    lengths = df[column].str.len()
    word_counts = df[column].str.split().str.len()
    
    return {
        'char_length': {
            'mean': lengths.mean(),
            'std': lengths.std(),
            'min': lengths.min(),
            'max': lengths.max(),
            'median': lengths.median()
        },
        'word_count': {
            'mean': word_counts.mean(),
            'std': word_counts.std(),
            'min': word_counts.min(),
            'max': word_counts.max(),
            'median': word_counts.median()
        }
    }
