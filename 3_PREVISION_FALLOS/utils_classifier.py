"""
Utilidades para clasificación de fallas en dispositivos.
Predicción de Mantenimiento Predictivo.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from typing import Dict, List, Tuple, Any, Optional

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, f1_score,
    precision_score, recall_score, average_precision_score,
    accuracy_score, balanced_accuracy_score
)
from sklearn.calibration import calibration_curve

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek

import optuna
from optuna.samplers import TPESampler

import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# CONSTANTES
# =============================================================================
COST_FAILURE = 1.0  # Costo si el dispositivo falla
COST_MAINTENANCE = 0.5  # Costo de mantenimiento preventivo
ATTRIBUTES = [f'attribute{i}' for i in range(1, 10)]

# Colores para gráficos
COLORS = {
    'primary': '#2196F3',
    'success': '#4CAF50',
    'danger': '#F44336',
    'warning': '#FF9800',
    'info': '#00BCD4',
    'dark': '#37474F'
}


# =============================================================================
# FUNCIONES GENÉRICAS UNIFICADAS
# =============================================================================

def load_data(data_path: str, sort_by_date: bool = True) -> pd.DataFrame:
    """
    Carga el dataset de dispositivos (función unificada).
    
    Args:
        data_path: Ruta al archivo CSV
        sort_by_date: Si True, ordena por fecha
        
    Returns:
        DataFrame con los datos
    """
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    if sort_by_date:
        df = df.sort_values('date').reset_index(drop=True)
    return df


def temporal_split(
    df: pd.DataFrame, 
    train_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """
    Realiza split temporal de los datos (función unificada).
    
    Args:
        df: DataFrame ordenado por fecha
        train_ratio: Proporción para entrenamiento
    
    Returns:
        Tuple de (df_train, df_test, fecha_corte)
    """
    train_size = int(len(df) * train_ratio)
    split_date = df.iloc[train_size]['date']
    
    df_train = df.iloc[:train_size].copy()
    df_test = df.iloc[train_size:].copy()
    
    return df_train, df_test, split_date


def predict_with_threshold(
    model, 
    X: np.ndarray, 
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera predicciones con un threshold personalizado (función unificada).
    
    Args:
        model: Modelo entrenado con predict_proba
        X: Features
        threshold: Umbral de decisión
        
    Returns:
        Tuple de (probabilidades, predicciones binarias)
    """
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    return y_proba, y_pred


def get_feature_columns(df: pd.DataFrame, exclude: List[str] = None) -> List[str]:
    """
    Obtiene las columnas de features excluyendo las especificadas.
    
    Args:
        df: DataFrame
        exclude: Columnas a excluir (default: ['date', 'device', 'failure'])
    
    Returns:
        Lista de nombres de columnas de features
    """
    if exclude is None:
        exclude = ['date', 'device', 'failure']
    return [c for c in df.columns if c not in exclude]


def plot_confusion_matrix_with_costs(
    metrics: Dict[str, Any], 
    window_days: int = None,
    title_suffix: str = '',
    cost_failure: float = COST_FAILURE,
    cost_maintenance: float = COST_MAINTENANCE
) -> None:
    """
    Visualiza matriz de confusión con costos (función unificada).
    
    Args:
        metrics: Dict con 'tp', 'fp', 'fn', 'cost', 'savings_pct'
        window_days: Días de ventana (para el título)
        title_suffix: Sufijo adicional para el título
        cost_failure: Costo de falla
        cost_maintenance: Costo de mantenimiento
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    labels = np.array([
        [f"TN\n(no aplica)", f"FP\n{metrics['fp']}\nCosto: ${metrics['fp']*cost_maintenance:.1f}"],
        [f"FN\n{metrics['fn']}\nCosto: ${metrics['fn']*cost_failure:.1f}", 
         f"TP\n{metrics['tp']}\nCosto: ${metrics['tp']*cost_maintenance:.1f}"]
    ])
    
    colors = np.array([[0.9, 0.3], [0.3, 0.9]])
    ax.imshow(colors, cmap='RdYlGn', vmin=0, vmax=1)
    
    for i in range(2):
        for j in range(2):
            ax.text(j, i, labels[i, j], ha='center', va='center', 
                    fontsize=12, fontweight='bold')
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicción: No Falla', 'Predicción: Falla'])
    ax.set_yticklabels(['Real: No Falla', 'Real: Falla'])
    
    title = 'Matriz de Confusión'
    if window_days:
        title += f' (Ventana {window_days} días)'
    title += f'\nCosto Total: ${metrics["cost"]:.1f} | Ahorro: {metrics["savings_pct"]:+.1f}%'
    if title_suffix:
        title += f' {title_suffix}'
    ax.set_title(title)
    
    plt.tight_layout()
    plt.show()


def plot_feature_importance_generic(
    model, 
    feature_cols: List[str], 
    top_n: int = 20,
    highlight_pattern: str = 'rule'
) -> None:
    """
    Visualiza importancia de features (función unificada).
    
    Args:
        model: Modelo con feature_importances_ o estimators_
        feature_cols: Nombres de features
        top_n: Número de features a mostrar
        highlight_pattern: Patrón para resaltar features (color diferente)
    """
    # Obtener importancias
    importances = None
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'estimators_'):
        # Para BalancedBaggingClassifier y similares
        importances = np.zeros(len(feature_cols))
        count = 0
        for idx, est in enumerate(model.estimators_):
            # Los estimadores pueden ser pipelines o tener estructura anidada
            base_est = est
            # Buscar el estimador base si está envuelto
            if hasattr(est, 'steps'):  # Pipeline
                base_est = est.steps[-1][1]
            elif hasattr(est, 'estimator'):
                base_est = est.estimator
            
            if hasattr(base_est, 'feature_importances_'):
                # Ajustar por features_indices_ si existe (bagging usa subconjuntos)
                if hasattr(model, 'estimators_features_'):
                    feat_indices = model.estimators_features_[idx]
                    temp_imp = np.zeros(len(feature_cols))
                    temp_imp[feat_indices] = base_est.feature_importances_
                    importances += temp_imp
                else:
                    importances += base_est.feature_importances_
                count += 1
        if count > 0:
            importances /= count
        else:
            importances = None
    
    if importances is None or np.all(importances == 0):
        print("El modelo no tiene feature_importances_ o todas son cero")
        print("Intentando usar permutation importance (puede tardar)...")
        # Fallback: mostrar mensaje informativo
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, 
                'Feature importance no disponible para este modelo.\n\n'
                'BalancedBaggingClassifier no expone feature_importances_ directamente.\n'
                'Considera usar sklearn.inspection.permutation_importance\n'
                'para calcular la importancia de features.',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, wrap=True)
        ax.set_title(f'Top {top_n} Features más Importantes')
        ax.axis('off')
        plt.tight_layout()
        plt.show()
        return
    
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    top_features = importance_df.tail(top_n)
    colors_imp = [COLORS['primary'] if highlight_pattern not in f else COLORS['warning'] 
                  for f in top_features['feature']]
    
    ax.barh(top_features['feature'], top_features['importance'], color=colors_imp)
    ax.set_xlabel('Importancia')
    ax.set_title(f'Top {top_n} Features más Importantes')
    
    plt.tight_layout()
    plt.show()


def plot_threshold_analysis_generic(
    df_thresh: pd.DataFrame, 
    baseline: float,
    current_threshold: float = 0.5,
    current_cost: float = None
) -> None:
    """
    Visualiza análisis de thresholds (función unificada).
    
    Args:
        df_thresh: DataFrame con columnas threshold, cost, precision, recall, f1, tp, fp, savings_pct
        baseline: Costo baseline
        current_threshold: Threshold actual del modelo
        current_cost: Costo actual (opcional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Costo vs Threshold
    ax1 = axes[0, 0]
    ax1.plot(df_thresh['threshold'], df_thresh['cost'], color=COLORS['primary'], linewidth=2)
    ax1.axhline(y=baseline, color=COLORS['danger'], linestyle='--', label='Baseline')
    ax1.axvline(x=current_threshold, color=COLORS['success'], linestyle='--', 
                label=f'Threshold = {current_threshold}')
    if current_cost:
        ax1.scatter([current_threshold], [current_cost], color=COLORS['success'], s=100, zorder=5)
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Costo ($)')
    ax1.set_title('Costo vs Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Precision/Recall vs Threshold
    ax2 = axes[0, 1]
    ax2.plot(df_thresh['threshold'], df_thresh['precision']*100, color=COLORS['success'], 
             linewidth=2, label='Precision')
    ax2.plot(df_thresh['threshold'], df_thresh['recall']*100, color=COLORS['danger'], 
             linewidth=2, label='Recall')
    ax2.plot(df_thresh['threshold'], df_thresh['f1']*100, color=COLORS['primary'], 
             linewidth=2, label='F1', linestyle='--')
    ax2.axvline(x=current_threshold, color=COLORS['dark'], linestyle='--', alpha=0.5)
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Porcentaje (%)')
    ax2.set_title('Métricas vs Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # TP/FP vs Threshold
    ax3 = axes[1, 0]
    ax3.plot(df_thresh['threshold'], df_thresh['tp'], color=COLORS['success'], 
             linewidth=2, label='True Positives')
    ax3.plot(df_thresh['threshold'], df_thresh['fp'], color=COLORS['danger'], 
             linewidth=2, label='False Positives')
    ax3.axvline(x=current_threshold, color=COLORS['dark'], linestyle='--', alpha=0.5)
    ax3.set_xlabel('Threshold')
    ax3.set_ylabel('Cantidad')
    ax3.set_title('TP y FP vs Threshold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Ahorro vs Threshold
    ax4 = axes[1, 1]
    colors_savings = [COLORS['success'] if s > 0 else COLORS['danger'] 
                      for s in df_thresh['savings_pct']]
    bar_width = df_thresh['threshold'].diff().mean() * 0.8 if len(df_thresh) > 1 else 0.04
    ax4.bar(df_thresh['threshold'], df_thresh['savings_pct'], color=colors_savings, width=bar_width)
    ax4.axhline(y=0, color='black', linewidth=1)
    ax4.axvline(x=current_threshold, color=COLORS['dark'], linestyle='--', alpha=0.5,
                label=f'Threshold = {current_threshold}')
    ax4.set_xlabel('Threshold')
    ax4.set_ylabel('Ahorro (%)')
    ax4.set_title('Ahorro vs Threshold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def display_threshold_table(
    df_thresh: pd.DataFrame, 
    current_threshold: float = None
) -> None:
    """
    Muestra tabla formateada de métricas por threshold (función unificada).
    """
    print("=" * 110)
    print(" TABLA DE MÉTRICAS POR THRESHOLD")
    print("=" * 110)
    
    df_display = df_thresh.copy()
    df_display['threshold'] = df_display['threshold'].apply(lambda x: f"{x:.2f}")
    df_display['precision'] = df_display['precision'].apply(lambda x: f"{x*100:.1f}%")
    df_display['recall'] = df_display['recall'].apply(lambda x: f"{x*100:.1f}%")
    df_display['f1'] = df_display['f1'].apply(lambda x: f"{x*100:.1f}%")
    df_display['cost'] = df_display['cost'].apply(lambda x: f"${x:.1f}")
    df_display['savings_pct'] = df_display['savings_pct'].apply(lambda x: f"{x:+.1f}%")
    
    # Renombrar columnas si existen
    rename_map = {
        'threshold': 'Threshold', 'tp': 'TP', 'fp': 'FP', 'fn': 'FN', 'tn': 'TN',
        'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1', 
        'cost': 'Costo', 'savings_pct': 'Ahorro'
    }
    df_display = df_display.rename(columns={k: v for k, v in rename_map.items() if k in df_display.columns})
    
    pd.set_option('display.max_rows', None)
    try:
        from IPython.display import display
        display(df_display)
    except ImportError:
        print(df_display.to_string())
    pd.reset_option('display.max_rows')
    
    if current_threshold:
        print(f"\n  Threshold actual del modelo: {current_threshold}")


# =============================================================================
# HELPERS DE IMPRESIÓN
# =============================================================================
def print_header(title: str, char: str = "=", width: int = 60) -> None:
    """Imprime un encabezado formateado."""
    print(char * width)
    print(title.center(width))
    print(char * width)


def print_section(title: str) -> None:
    """Imprime un título de sección."""
    print(f"\n{'='*60}")
    print(title)
    print("="*60)


# =============================================================================
# FUNCIONES DE COSTO
# =============================================================================
def calculate_cost(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """
    Calcula el costo total basado en la matriz de costos.
    
    Matriz de costos:
    - True Positive (predice falla, es falla): 0.5 (mantenimiento previene falla)
    - False Positive (predice falla, no es falla): 0.5 (mantenimiento innecesario)
    - False Negative (predice no falla, es falla): 1.0 (falla no prevenida)
    - True Negative (predice no falla, no es falla): 0.0 (óptimo)
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        
    Returns:
        Tuple con (costo_total, diccionario_desglose)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    cost_tp = tp * COST_MAINTENANCE  # Mantenimiento que previene falla
    cost_fp = fp * COST_MAINTENANCE  # Mantenimiento innecesario
    cost_fn = fn * COST_FAILURE      # Falla no prevenida
    cost_tn = tn * 0                 # Sin costo
    
    total_cost = cost_tp + cost_fp + cost_fn + cost_tn
    
    return total_cost, {'TP': cost_tp, 'FP': cost_fp, 'FN': cost_fn, 'TN': cost_tn}


def calculate_baseline_cost(y_true: np.ndarray) -> float:
    """Calcula el costo si no hacemos nada (todas las fallas ocurren)."""
    return y_true.sum() * COST_FAILURE


# =============================================================================
# MODELO BASELINE (SIN FEATURE ENGINEERING)
# =============================================================================
def train_baseline_model(
    df: pd.DataFrame,
    feature_cols: List[str] = None,
    target_col: str = 'failure',
    date_col: str = 'date',
    test_size: float = 0.2,
    use_smote: bool = True,
    smote_sampling_strategy: float = 0.3
) -> Dict[str, Any]:
    """
    Entrena modelos baseline con las features originales (sin feature engineering).
    Útil para establecer un punto de comparación antes de agregar features.
    
    Args:
        df: DataFrame con los datos
        feature_cols: Lista de columnas a usar como features
        target_col: Nombre de la columna objetivo
        date_col: Nombre de la columna de fecha
        test_size: Proporción de datos para test
        use_smote: Si True, aplica SMOTE para balancear clases
        smote_sampling_strategy: Ratio de oversampling para SMOTE
    
    Returns:
        Dict con resultados del modelo baseline
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score
    
    smote_str = "con SMOTE" if use_smote else "sin SMOTE"
    print_section(f"MODELO BASELINE (Features Originales - {smote_str})")
    
    # Usar ATTRIBUTES si no se especifican features
    if feature_cols is None:
        feature_cols = ATTRIBUTES
    
    print(f"\nFeatures utilizadas ({len(feature_cols)}): {feature_cols}")
    
    # Preparar datos
    df_model = df.copy()
    if date_col in df_model.columns:
        df_model[date_col] = pd.to_datetime(df_model[date_col])
        df_model = df_model.sort_values(date_col)
    
    X = df_model[feature_cols]
    y = df_model[target_col]
    
    # Split temporal (últimos test_size% para test)
    split_idx = int(len(df_model) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"\nSplit de datos:")
    print(f"  Train: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Test:  {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")
    print(f"  Fallas en train: {y_train.sum():,} ({y_train.mean()*100:.2f}%)")
    print(f"  Fallas en test:  {y_test.sum():,} ({y_test.mean()*100:.2f}%)")
    
    # Escalar features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Aplicar SMOTE si está habilitado
    if use_smote:
        smote = SMOTE(random_state=42, sampling_strategy=smote_sampling_strategy)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        print(f"\nSMOTE aplicado:")
        print(f"  Antes: {len(y_train):,} muestras ({y_train.sum():,} fallas)")
        print(f"  Después: {len(y_train_balanced):,} muestras ({y_train_balanced.sum():,} fallas)")
    else:
        X_train_balanced, y_train_balanced = X_train_scaled, y_train
    
    # Modelos a probar
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    # Calcular costos baseline
    baseline_cost = calculate_baseline_cost(y_test)
    optimal_cost = y_test.sum() * COST_MAINTENANCE
    
    print(f"\n{'='*70}")
    print(f"{'Modelo':<25} {'AUC-ROC':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Costo':>12}")
    print(f"{'='*70}")
    
    results = {}
    for name, model in models.items():
        model.fit(X_train_balanced, y_train_balanced)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        auc = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cost, _ = calculate_cost(y_test, y_pred)
        savings_pct = (baseline_cost - cost) / baseline_cost * 100
        
        results[name] = {
            'model': model,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cost': cost,
            'savings_pct': savings_pct,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"{name:<25} {auc:>10.4f} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} ${cost:>10,.0f}")
    
    print(f"{'='*70}")
    print(f"{'Baseline (no hacer nada)':<25} {'-':>10} {'-':>10} {'-':>10} {'-':>10} ${baseline_cost:>10,.0f}")
    print(f"{'Optimo (pred. perfecta)':<25} {'-':>10} {'-':>10} {'-':>10} {'-':>10} ${optimal_cost:>10,.0f}")
    
    # Mejor modelo (seleccionado por F1)
    best_name = max(results, key=lambda x: results[x]['f1'])
    best = results[best_name]
    
    print(f"\n{'='*70}")
    print(f"MEJOR MODELO BASELINE: {best_name} (por F1)")
    print(f"{'='*70}")
    print(f"  F1-Score: {best['f1']:.4f}")
    print(f"  Recall:   {best['recall']:.4f} (detecta {best['recall']*100:.1f}% de las fallas)")
    print(f"  Precision:{best['precision']:.4f}")
    print(f"  AUC-ROC:  {best['auc']:.4f}")
    print(f"  Costo:    ${best['cost']:,.0f}")
    print(f"  Ahorro:   {best['savings_pct']:.1f}% vs no hacer nada")
    print(f"\nNota: Este es el baseline con {len(feature_cols)} features originales.")
    print(f"El objetivo es mejorar estos resultados con feature engineering.")
    
    return {
        'results': results,
        'best_model_name': best_name,
        'best_model': best,
        'baseline_cost': baseline_cost,
        'optimal_cost': optimal_cost,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
class FeatureEngineer:
    """Clase para crear features para el modelo de mantenimiento predictivo."""
    
    def __init__(
        self,
        rolling_windows: List[int] = [3, 7, 14],
        attributes: List[str] = ATTRIBUTES
    ):
        self.rolling_windows = rolling_windows
        self.attributes = attributes
        
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features temporales básicas (diarias y semanales)."""
        df = df.copy()
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Encoding cíclico (solo día de la semana)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea rolling statistics (estadísticas móviles por dispositivo)."""
        df = df.copy()
        for attr in self.attributes:
            for window in self.rolling_windows:
                # Media móvil
                df[f'{attr}_rolling_mean_{window}'] = df.groupby('device')[attr].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                # Máximo móvil
                df[f'{attr}_rolling_max_{window}'] = df.groupby('device')[attr].transform(
                    lambda x: x.rolling(window=window, min_periods=1).max()
                )
        return df
    
    def create_diff_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features de diferencia (cambio respecto al día anterior)."""
        df = df.copy()
        for attr in self.attributes:
            df[f'{attr}_diff'] = df.groupby('device')[attr].diff()
            # pct_change puede generar inf cuando el valor anterior es 0
            # Usamos una versión segura: (actual - anterior) / (anterior + 1)
            prev_val = df.groupby('device')[attr].shift(1)
            df[f'{attr}_pct_change'] = (df[attr] - prev_val) / (prev_val.abs() + 1)
        return df
    
    def create_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features de anomalía (z-score móvil)."""
        df = df.copy()
        for attr in self.attributes:
            rolling_mean = df.groupby('device')[attr].transform(
                lambda x: x.rolling(window=7, min_periods=1).mean()
            )
            rolling_std = df.groupby('device')[attr].transform(
                lambda x: x.rolling(window=7, min_periods=1).std()
            )
            df[f'{attr}_zscore_7'] = (df[attr] - rolling_mean) / (rolling_std + 1e-8)
        return df
    
    def create_device_aggregate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea features agregadas por dispositivo usando solo datos pasados.
        Usa expanding window con shift(1) para evitar data leakage.
        """
        df = df.copy()
        df = df.sort_values(['device', 'date']).reset_index(drop=True)
        
        for attr in self.attributes:
            # Media expandida usando solo datos pasados (shift para no incluir día actual)
            df[f'{attr}_device_mean'] = df.groupby('device')[attr].transform(
                lambda x: x.expanding().mean().shift(1)
            )
            # Desviación estándar expandida usando solo datos pasados
            df[f'{attr}_device_std'] = df.groupby('device')[attr].transform(
                lambda x: x.expanding().std().shift(1)
            )
            # Máximo expandido usando solo datos pasados
            df[f'{attr}_device_max'] = df.groupby('device')[attr].transform(
                lambda x: x.expanding().max().shift(1)
            )
            # Mínimo expandido usando solo datos pasados
            df[f'{attr}_device_min'] = df.groupby('device')[attr].transform(
                lambda x: x.expanding().min().shift(1)
            )
        
        return df
    
    def create_device_age_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea feature de antigüedad del dispositivo."""
        df = df.copy()
        device_first_date = df.groupby('device')['date'].min().reset_index()
        device_first_date.columns = ['device', 'first_date']
        df = df.merge(device_first_date, on='device', how='left')
        df['device_age_days'] = (df['date'] - df['first_date']).dt.days
        df = df.drop('first_date', axis=1)
        return df
    
    def create_device_failure_history_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea features basadas en el historial de fallas del dispositivo.
        Usa solo información pasada para evitar data leakage.
        """
        df = df.copy()
        
        # Asegurar orden por device y fecha
        df = df.sort_values(['device', 'date']).reset_index(drop=True)
        
        # Fallas acumuladas ANTES del día actual (shift para no incluir el día actual)
        df['device_failures_so_far'] = df.groupby('device')['failure'].cumsum().shift(1)
        df['device_failures_so_far'] = df.groupby('device')['device_failures_so_far'].ffill().fillna(0)
        
        # Días de operación hasta ese momento
        df['device_days_so_far'] = df.groupby('device').cumcount()
        
        # Tasa de fallas hasta ese momento (evitar división por 0)
        df['device_failure_rate_so_far'] = df['device_failures_so_far'] / (df['device_days_so_far'] + 1)
        
        # Flag: ¿el dispositivo ha fallado antes?
        df['device_has_failed_before'] = (df['device_failures_so_far'] > 0).astype(int)
        
        return df
    
    def create_device_prefix_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features basadas en el prefijo del dispositivo."""
        df = df.copy()
        df['device_prefix'] = df['device'].str[:3]
        device_prefix_dummies = pd.get_dummies(df['device_prefix'], prefix='prefix')
        df = pd.concat([df, device_prefix_dummies], axis=1)
        df = df.drop('device_prefix', axis=1)
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features de interacción entre atributos."""
        df = df.copy()
        df['sum_all_attributes'] = df[self.attributes].sum(axis=1)
        df['mean_all_attributes'] = df[self.attributes].mean(axis=1)
        df['std_all_attributes'] = df[self.attributes].std(axis=1)
        df['max_all_attributes'] = df[self.attributes].max(axis=1)
        df['min_all_attributes'] = df[self.attributes].min(axis=1)
        df['range_all_attributes'] = df['max_all_attributes'] - df['min_all_attributes']
        df['attributes_nonzero_count'] = (df[self.attributes] > 0).sum(axis=1)
        
        # Features específicas
        df['attr7_attr8_equal'] = (df['attribute7'] == df['attribute8']).astype(int)
        df['attr7_is_active'] = (df['attribute7'] > 0).astype(int)
        df['attr8_is_active'] = (df['attribute8'] > 0).astype(int)
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica todas las transformaciones de feature engineering."""
        df = df.sort_values(['device', 'date']).reset_index(drop=True)
        
        df = self.create_temporal_features(df)
        df = self.create_rolling_features(df)
        df = self.create_diff_features(df)
        df = self.create_anomaly_features(df)
        df = self.create_device_aggregate_features(df)
        df = self.create_device_age_feature(df)
        df = self.create_device_failure_history_features(df)
        df = self.create_device_prefix_features(df)
        df = self.create_interaction_features(df)
        
        return df


# =============================================================================
# PREPARACIÓN DE DATOS
# =============================================================================
class DataPreparator:
    """Clase para preparar datos para el modelo."""
    
    def __init__(self, exclude_cols: List[str] = ['date', 'device', 'failure']):
        self.exclude_cols = exclude_cols
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = RobustScaler()
        self.feature_cols = None
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepara features y target."""
        # Remover filas con nulos (generados por lag/rolling features)
        rows_before = len(df)
        df_clean = df.dropna()
        rows_after = len(df_clean)
        rows_dropped = rows_before - rows_after
        
        print(f"Filas dropeadas (NaN): {rows_dropped:,} ({rows_dropped/rows_before*100:.1f}%)")
        print(f"Filas restantes: {rows_after:,} ({rows_after/rows_before*100:.1f}%)")
        
        # Definir features y target
        self.feature_cols = [col for col in df_clean.columns if col not in self.exclude_cols]
        X = df_clean[self.feature_cols]
        y = df_clean['failure']
        
        return X, y, df_clean
    
    def fit_transform_imputer(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aplica imputación de valores nulos e infinitos."""
        # Primero reemplazar infinitos por NaN
        X_clean = X.replace([np.inf, -np.inf], np.nan)
        
        # Luego aplicar imputación
        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X_clean),
            columns=X.columns,
            index=X.index
        )
        
        # Verificar si quedaron NaN (por columnas con todos NaN)
        X_imputed = X_imputed.fillna(0)
        return X_imputed
    
    def transform_imputer(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aplica imputación a nuevos datos."""
        # Primero reemplazar infinitos por NaN
        X_clean = X.replace([np.inf, -np.inf], np.nan)
        
        # Luego aplicar imputación
        X_imputed = pd.DataFrame(
            self.imputer.transform(X_clean),
            columns=X.columns,
            index=X.index
        )
        
        # Verificar si quedaron NaN
        X_imputed = X_imputed.fillna(0)
        return X_imputed
    
    def temporal_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        df: pd.DataFrame,
        train_ratio: float = 0.8
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Realiza split temporal respetando el orden cronológico."""
        dates_sorted = df['date'].sort_values()
        split_date = dates_sorted.quantile(train_ratio)
        
        train_mask = df['date'] < split_date
        test_mask = df['date'] >= split_date
        
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        
        return X_train, X_test, y_train, y_test, split_date
    
    def select_features_by_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        importance_threshold: float = 0.001,
        top_k: int = None
    ) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
        """
        Selecciona features basándose en importancia usando un modelo rápido de Random Forest.
        
        Args:
            X: DataFrame con features
            y: Serie con target
            importance_threshold: Umbral mínimo de importancia (default 0.001)
            top_k: Si se especifica, selecciona las top_k features más importantes
            
        Returns:
            X_selected: DataFrame con features seleccionadas
            selected_features: Lista de nombres de features seleccionadas
            importance_df: DataFrame con importancias de todas las features
        """
        from sklearn.ensemble import RandomForestClassifier
        
        print("\n🔍 Selección de features por importancia...")
        
        # Entrenar modelo rápido para obtener importancias
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # Crear DataFrame de importancias
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Seleccionar features
        if top_k is not None:
            selected_features = importance_df.head(top_k)['feature'].tolist()
            print(f"   Seleccionadas top {top_k} features")
        else:
            selected_features = importance_df[importance_df['importance'] >= importance_threshold]['feature'].tolist()
            print(f"   Umbral de importancia: {importance_threshold}")
        
        # Features eliminadas
        removed_features = [f for f in X.columns if f not in selected_features]
        
        print(f"   Features originales: {len(X.columns)}")
        print(f"   Features seleccionadas: {len(selected_features)}")
        print(f"   Features eliminadas: {len(removed_features)}")
        
        if removed_features:
            print(f"\n   ❌ Features eliminadas (baja importancia):")
            for feat in removed_features[:20]:  # Mostrar máximo 20
                imp = importance_df[importance_df['feature'] == feat]['importance'].values[0]
                print(f"      - {feat}: {imp:.6f}")
            if len(removed_features) > 20:
                print(f"      ... y {len(removed_features) - 20} más")
        
        # Actualizar feature_cols
        self.feature_cols = selected_features
        X_selected = X[selected_features]
        
        return X_selected, selected_features, importance_df
    
    def remove_low_variance_features(
        self,
        X: pd.DataFrame,
        variance_threshold: float = 0.01
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Elimina features con varianza muy baja (casi constantes).
        
        Args:
            X: DataFrame con features
            variance_threshold: Umbral mínimo de varianza normalizada
            
        Returns:
            X_filtered: DataFrame sin features de baja varianza
            removed_features: Lista de features eliminadas
        """
        from sklearn.preprocessing import StandardScaler
        
        print("\n🔍 Eliminando features de baja varianza...")
        
        # Escalar para normalizar varianzas
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        variances = np.var(X_scaled, axis=0)
        
        # Identificar features de baja varianza
        low_var_mask = variances < variance_threshold
        removed_features = X.columns[low_var_mask].tolist()
        kept_features = X.columns[~low_var_mask].tolist()
        
        print(f"   Umbral de varianza: {variance_threshold}")
        print(f"   Features eliminadas: {len(removed_features)}")
        
        if removed_features:
            print(f"\n   ❌ Features eliminadas (baja varianza):")
            for feat in removed_features[:10]:
                var = variances[X.columns.tolist().index(feat)]
                print(f"      - {feat}: var={var:.6f}")
            if len(removed_features) > 10:
                print(f"      ... y {len(removed_features) - 10} más")
        
        # Actualizar feature_cols
        self.feature_cols = kept_features
        
        return X[kept_features], removed_features
    
    def fit_transform_scaler(self, X: np.ndarray) -> np.ndarray:
        """Aplica escalado a los datos."""
        return self.scaler.fit_transform(X)
    
    def transform_scaler(self, X: np.ndarray) -> np.ndarray:
        """Aplica escalado a nuevos datos."""
        return self.scaler.transform(X)


# =============================================================================
# MODELOS
# =============================================================================
class ModelTrainer:
    """Clase para entrenar y evaluar modelos."""
    
    def __init__(self, class_weight_ratio: float):
        self.class_weight_ratio = class_weight_ratio
        self.models = self._define_models()
        self.results = {}
        
    def _define_models(self) -> Dict[str, Any]:
        """Define los modelos a entrenar."""
        return {
            'Logistic Regression': LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                max_depth=15,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                scale_pos_weight=self.class_weight_ratio,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            ),
            'LightGBM': LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                scale_pos_weight=self.class_weight_ratio,
                random_state=42,
                verbose=-1,
                force_col_wise=True
            ),
            'CatBoost': CatBoostClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                scale_pos_weight=self.class_weight_ratio,
                random_state=42,
                verbose=0,
                allow_writing_files=False
            )
        }
    
    def apply_resampling(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        method: str = 'smote_enn',
        sampling_strategy: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aplica técnicas de resampling para balancear clases.
        
        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            method: Método de resampling. Opciones:
                - 'smote': SMOTE clásico
                - 'smote_enn': SMOTE + Edited Nearest Neighbors (limpia bordes)
                - 'smote_tomek': SMOTE + Tomek Links (elimina ejemplos ambiguos)
                - 'adasyn': Adaptive Synthetic Sampling (más muestras en zonas difíciles)
            sampling_strategy: Ratio de oversampling para SMOTE/ADASYN
            
        Returns:
            Tuple con (X_balanced, y_balanced)
        """
        print(f"\n🔄 Aplicando resampling: {method.upper()}")
        print(f"   Antes: {len(y_train):,} muestras ({y_train.sum():,} fallas, {y_train.mean()*100:.2f}%)")
        
        if method == 'smote':
            resampler = SMOTE(random_state=42, sampling_strategy=sampling_strategy)
        elif method == 'smote_enn':
            # SMOTE + ENN: genera sintéticos y luego limpia ejemplos mal clasificados
            resampler = SMOTEENN(
                smote=SMOTE(random_state=42, sampling_strategy=sampling_strategy),
                random_state=42
            )
        elif method == 'smote_tomek':
            # SMOTE + Tomek: genera sintéticos y elimina pares Tomek (ejemplos ambiguos)
            resampler = SMOTETomek(
                smote=SMOTE(random_state=42, sampling_strategy=sampling_strategy),
                random_state=42
            )
        elif method == 'adasyn':
            # ADASYN: genera más muestras en regiones difíciles de aprender
            resampler = ADASYN(random_state=42, sampling_strategy=sampling_strategy)
        else:
            raise ValueError(f"Método '{method}' no soportado. Use: smote, smote_enn, smote_tomek, adasyn")
        
        X_balanced, y_balanced = resampler.fit_resample(X_train, y_train)
        
        print(f"   Después: {len(y_balanced):,} muestras ({y_balanced.sum():,} fallas, {y_balanced.mean()*100:.2f}%)")
        
        return X_balanced, y_balanced
    
    # Mantener compatibilidad con código anterior
    def apply_smote(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sampling_strategy: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Aplica SMOTE para balancear clases (legacy, usar apply_resampling)."""
        return self.apply_resampling(X_train, y_train, method='smote', sampling_strategy=sampling_strategy)
    
    def train_and_evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        baseline_cost: float,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Entrena y evalúa todos los modelos."""
        for name, model in self.models.items():
            if verbose:
                print(f"\nEntrenando: {name}")
            
            # Entrenar
            model.fit(X_train, y_train)
            
            # Predicciones
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Métricas
            auc_score = roc_auc_score(y_test, y_pred_proba)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            avg_precision = average_precision_score(y_test, y_pred_proba)
            
            # Calcular costo
            total_cost, cost_breakdown = calculate_cost(y_test, y_pred)
            savings = baseline_cost - total_cost
            savings_pct = (savings / baseline_cost) * 100
            
            self.results[name] = {
                'model': model,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'auc': auc_score,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'avg_precision': avg_precision,
                'total_cost': total_cost,
                'cost_breakdown': cost_breakdown,
                'savings': savings,
                'savings_pct': savings_pct
            }
            
            if verbose:
                print(f"  AUC-ROC: {auc_score:.4f}, Recall: {recall:.4f}, Ahorro: {savings_pct:.2f}%")
        
        return self.results
    
    def get_best_model(self, metric: str = 'f1') -> Tuple[str, Dict[str, Any]]:
        """Obtiene el mejor modelo según la métrica especificada."""
        best_name = max(self.results, key=lambda x: self.results[x][metric])
        return best_name, self.results[best_name]
    
    def get_comparison_df(self) -> pd.DataFrame:
        """Retorna DataFrame con comparación de modelos."""
        return pd.DataFrame({
            'Modelo': list(self.results.keys()),
            'AUC-ROC': [self.results[m]['auc'] for m in self.results],
            'Precision': [self.results[m]['precision'] for m in self.results],
            'Recall': [self.results[m]['recall'] for m in self.results],
            'F1-Score': [self.results[m]['f1'] for m in self.results],
            'Costo Total': [self.results[m]['total_cost'] for m in self.results],
            'Ahorro (%)': [self.results[m]['savings_pct'] for m in self.results]
        }).round(4).sort_values('Ahorro (%)', ascending=False)


# =============================================================================
# OPTIMIZACIÓN CON OPTUNA
# =============================================================================
class OptunaOptimizer:
    """Clase para optimizar hiperparámetros con Optuna."""
    
    def __init__(
        self,
        model_type: str,
        class_weight_ratio: float,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        optimize_by: str = 'cost',
        baseline_cost: float = None,
        use_early_stopping: bool = True,
        early_stopping_rounds: int = 50,
        validation_fraction: float = 0.15
    ):
        """
        Inicializa el optimizador.
        
        Args:
            model_type: Tipo de modelo ('XGBoost', 'LightGBM', 'CatBoost')
            class_weight_ratio: Ratio de peso de clases
            X_train, y_train: Datos de entrenamiento
            X_test, y_test: Datos de test
            optimize_by: Métrica a optimizar ('cost', 'f1', 'recall', 'auc')
            baseline_cost: Costo baseline (requerido si optimize_by='cost')
            use_early_stopping: Si True, usa early stopping durante el entrenamiento
            early_stopping_rounds: Número de rondas sin mejora para detener
            validation_fraction: Fracción de datos de train para validación de early stopping
        """
        self.model_type = model_type
        self.class_weight_ratio = class_weight_ratio
        self.X_test = X_test
        self.y_test = y_test
        self.optimize_by = optimize_by
        self.baseline_cost = baseline_cost
        self.use_early_stopping = use_early_stopping
        self.early_stopping_rounds = early_stopping_rounds
        self.study = None
        self.best_model = None
        self.best_threshold = 0.5
        
        # Crear conjunto de validación para early stopping
        if use_early_stopping and validation_fraction > 0:
            from sklearn.model_selection import train_test_split
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                X_train, y_train,
                test_size=validation_fraction,
                random_state=42,
                stratify=y_train
            )
            print(f"📊 Early Stopping habilitado:")
            print(f"   - Rounds sin mejora: {early_stopping_rounds}")
            print(f"   - Train: {len(self.X_train):,} | Validation: {len(self.X_val):,}")
        else:
            self.X_train = X_train
            self.y_train = y_train
            self.X_val = None
            self.y_val = None
        
        # Validar parámetros
        valid_metrics = ['cost', 'f1', 'recall', 'auc']
        if optimize_by not in valid_metrics:
            raise ValueError(f"optimize_by debe ser uno de: {valid_metrics}")
        if optimize_by == 'cost' and baseline_cost is None:
            raise ValueError("baseline_cost es requerido cuando optimize_by='cost'")
    
    def _create_model(self, trial: optuna.Trial) -> Any:
        """Crea el modelo con los hiperparámetros sugeridos por el trial."""
        # Con early stopping permitimos más estimadores (el modelo parará antes si es necesario)
        max_estimators = 500 if self.use_early_stopping else 300
        
        if self.model_type == 'XGBoost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, max_estimators),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'scale_pos_weight': self.class_weight_ratio,
                'random_state': 42,
                'use_label_encoder': False,
                'eval_metric': 'logloss',
                'early_stopping_rounds': self.early_stopping_rounds if self.use_early_stopping else None
            }
            return XGBClassifier(**params)
        elif self.model_type == 'LightGBM':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, max_estimators),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'scale_pos_weight': self.class_weight_ratio,
                'random_state': 42,
                'verbose': -1,
                'force_col_wise': True
            }
            return LGBMClassifier(**params)
        else:  # CatBoost
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, max_estimators),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
                'random_strength': trial.suggest_float('random_strength', 0, 10),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'scale_pos_weight': self.class_weight_ratio,
                'random_state': 42,
                'verbose': 0,
                'allow_writing_files': False,
                'early_stopping_rounds': self.early_stopping_rounds if self.use_early_stopping else None
            }
            return CatBoostClassifier(**params)
    
    def _find_optimal_threshold_for_cost(self, y_pred_proba: np.ndarray) -> Tuple[float, float]:
        """
        Encuentra el umbral óptimo que minimiza el costo.
        
        Returns:
            Tuple con (umbral_optimo, costo_minimo)
        """
        best_cost = float('inf')
        best_threshold = 0.5
        
        for thresh in np.arange(0.05, 0.70, 0.02):
            y_pred = (y_pred_proba >= thresh).astype(int)
            cost, _ = calculate_cost(self.y_test, y_pred)
            if cost < best_cost:
                best_cost = cost
                best_threshold = thresh
        
        return best_threshold, best_cost
        
    def _fit_model_with_early_stopping(self, model: Any) -> Any:
        """
        Entrena el modelo con o sin early stopping según la configuración.
        
        Args:
            model: Modelo a entrenar
            
        Returns:
            Modelo entrenado
        """
        if self.use_early_stopping and self.X_val is not None:
            # Entrenar con early stopping
            if self.model_type == 'XGBoost':
                model.fit(
                    self.X_train, self.y_train,
                    eval_set=[(self.X_val, self.y_val)],
                    verbose=False
                )
            elif self.model_type == 'LightGBM':
                from lightgbm import early_stopping, log_evaluation
                model.fit(
                    self.X_train, self.y_train,
                    eval_set=[(self.X_val, self.y_val)],
                    callbacks=[
                        early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=False),
                        log_evaluation(period=0)  # Silencia logs
                    ]
                )
            else:  # CatBoost
                model.fit(
                    self.X_train, self.y_train,
                    eval_set=(self.X_val, self.y_val),
                    verbose=False
                )
        else:
            # Entrenar sin early stopping
            model.fit(self.X_train, self.y_train)
        
        return model
        
    def _objective(self, trial: optuna.Trial) -> float:
        """Función objetivo para Optuna."""
        model = self._create_model(trial)
        model = self._fit_model_with_early_stopping(model)
        
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Guardar el número de iteraciones usadas (útil para early stopping)
        if hasattr(model, 'best_iteration_'):
            trial.set_user_attr('best_iteration', model.best_iteration_)
        elif hasattr(model, 'best_iteration'):
            trial.set_user_attr('best_iteration', model.best_iteration)
        
        if self.optimize_by == 'cost':
            # Optimización por costo: también optimiza el umbral dentro de cada trial
            optimal_thresh, min_cost = self._find_optimal_threshold_for_cost(y_pred_proba)
            # Guardamos el umbral óptimo en el trial para recuperarlo después
            trial.set_user_attr('optimal_threshold', optimal_thresh)
            # Retornamos el costo negativo porque Optuna maximiza por defecto
            # y queremos minimizar el costo
            return -min_cost
        
        elif self.optimize_by == 'f1':
            y_pred = model.predict(self.X_test)
            return f1_score(self.y_test, y_pred)
        
        elif self.optimize_by == 'recall':
            y_pred = model.predict(self.X_test)
            return recall_score(self.y_test, y_pred)
        
        elif self.optimize_by == 'auc':
            return roc_auc_score(self.y_test, y_pred_proba)
    
    def optimize(self, n_trials: int = 50) -> optuna.Study:
        """Ejecuta la optimización."""
        sampler = TPESampler(seed=42)
        
        # Para costo queremos minimizar, pero usamos -costo así que maximizamos
        direction = 'maximize'
        
        self.study = optuna.create_study(direction=direction, sampler=sampler)
        
        print(f"\n🎯 Optimizando por: {self.optimize_by.upper()}")
        if self.optimize_by == 'cost':
            print(f"   (también optimiza el umbral de decisión en cada trial)")
        
        self.study.optimize(self._objective, n_trials=n_trials, show_progress_bar=True)
        
        # Recuperar el umbral óptimo del mejor trial (si optimizamos por costo)
        if self.optimize_by == 'cost':
            self.best_threshold = self.study.best_trial.user_attrs.get('optimal_threshold', 0.5)
            print(f"\n   ✅ Mejor costo encontrado: ${-self.study.best_value:,.0f}")
            print(f"   ✅ Umbral óptimo: {self.best_threshold:.2f}")
        else:
            print(f"\n   ✅ Mejor {self.optimize_by}: {self.study.best_value:.4f}")
        
        return self.study
    
    def get_best_model(self, use_best_iteration: bool = True) -> Tuple[Any, float]:
        """
        Retorna el mejor modelo con los parámetros optimizados y el umbral óptimo.
        
        Args:
            use_best_iteration: Si True y se usó early stopping, limita n_estimators
                               al mejor número de iteraciones encontrado
        
        Returns:
            Tuple con (modelo_entrenado, umbral_optimo)
        """
        if self.study is None:
            raise ValueError("Debe ejecutar optimize() primero")
        
        # Obtener parámetros del mejor trial
        best_params = self.study.best_params.copy()
        
        # Si usamos early stopping y queremos usar la mejor iteración
        best_iteration = self.study.best_trial.user_attrs.get('best_iteration')
        if use_best_iteration and best_iteration is not None:
            # Limitar n_estimators al mejor número de iteraciones (+ margen)
            best_params['n_estimators'] = min(best_params['n_estimators'], best_iteration + 10)
            print(f"   📉 Usando best_iteration: {best_iteration} (n_estimators ajustado a {best_params['n_estimators']})")
            
        if self.model_type == 'XGBoost':
            self.best_model = XGBClassifier(
                **best_params,
                scale_pos_weight=self.class_weight_ratio,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                early_stopping_rounds=self.early_stopping_rounds if self.use_early_stopping else None
            )
        elif self.model_type == 'LightGBM':
            self.best_model = LGBMClassifier(
                **best_params,
                scale_pos_weight=self.class_weight_ratio,
                random_state=42,
                verbose=-1,
                force_col_wise=True
            )
        else:  # CatBoost
            self.best_model = CatBoostClassifier(
                **best_params,
                scale_pos_weight=self.class_weight_ratio,
                random_state=42,
                verbose=0,
                allow_writing_files=False,
                early_stopping_rounds=self.early_stopping_rounds if self.use_early_stopping else None
            )
        
        # Entrenar con o sin early stopping
        self.best_model = self._fit_model_with_early_stopping(self.best_model)
        return self.best_model, self.best_threshold
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Retorna un resumen de la optimización realizada.
        
        Returns:
            Dict con información del mejor trial y métricas
        """
        if self.study is None:
            raise ValueError("Debe ejecutar optimize() primero")
        
        summary = {
            'optimize_by': self.optimize_by,
            'best_params': self.study.best_params,
            'best_threshold': self.best_threshold,
            'n_trials': len(self.study.trials),
            'model_type': self.model_type,
            'early_stopping_enabled': self.use_early_stopping,
            'early_stopping_rounds': self.early_stopping_rounds if self.use_early_stopping else None
        }
        
        # Agregar información de early stopping del mejor trial
        best_iteration = self.study.best_trial.user_attrs.get('best_iteration')
        if best_iteration is not None:
            summary['best_iteration'] = best_iteration
            summary['n_estimators_requested'] = self.study.best_params.get('n_estimators')
            summary['early_stop_occurred'] = best_iteration < self.study.best_params.get('n_estimators', 0)
        
        if self.optimize_by == 'cost':
            summary['best_cost'] = -self.study.best_value
            summary['savings_vs_baseline'] = self.baseline_cost - (-self.study.best_value)
            summary['savings_pct'] = (summary['savings_vs_baseline'] / self.baseline_cost) * 100
        else:
            summary['best_score'] = self.study.best_value
        
        return summary


# =============================================================================
# OPTIMIZACIÓN DE UMBRAL
# =============================================================================
def optimize_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    baseline_cost: float,
    thresholds: np.ndarray = np.arange(0.05, 0.95, 0.01),
    optimize_by: str = 'f1'
) -> Tuple[float, pd.DataFrame]:
    """
    Optimiza el umbral de decisión según la métrica especificada.
    
    Args:
        y_true: Valores reales
        y_pred_proba: Probabilidades predichas
        baseline_cost: Costo baseline (sin modelo)
        thresholds: Array de umbrales a evaluar
        optimize_by: Métrica a optimizar ('f1', 'recall', 'precision', 'cost')
        
    Returns:
        Tuple con (umbral_óptimo, DataFrame_resultados)
    """
    valid_metrics = ['f1', 'recall', 'precision', 'cost']
    if optimize_by not in valid_metrics:
        raise ValueError(f"optimize_by debe ser uno de: {valid_metrics}")
    
    results = []
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        total_cost, _ = calculate_cost(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        results.append({
            'threshold': thresh,
            'cost': total_cost,
            'savings': baseline_cost - total_cost,
            'precision': prec,
            'recall': rec,
            'f1': f1
        })
    
    df_results = pd.DataFrame(results)
    
    # Seleccionar umbral óptimo según métrica
    if optimize_by == 'cost':
        optimal_idx = df_results['cost'].idxmin()
    else:
        # Para f1, recall, precision queremos maximizar
        optimal_idx = df_results[optimize_by].idxmax()
    
    optimal_threshold = df_results.loc[optimal_idx, 'threshold']
    
    print(f"\n📊 Optimización por: {optimize_by.upper()}")
    print(f"   Umbral óptimo: {optimal_threshold:.2f}")
    print(f"   Métricas en umbral óptimo:")
    print(f"     - F1:        {df_results.loc[optimal_idx, 'f1']:.4f}")
    print(f"     - Recall:    {df_results.loc[optimal_idx, 'recall']:.4f}")
    print(f"     - Precision: {df_results.loc[optimal_idx, 'precision']:.4f}")
    print(f"     - Costo:     ${df_results.loc[optimal_idx, 'cost']:,.2f}")
    
    return optimal_threshold, df_results


# =============================================================================
# VISUALIZACIÓN
# =============================================================================
class Visualizer:
    """Clase para crear visualizaciones."""
    
    @staticmethod
    def plot_failure_distribution(df: pd.DataFrame) -> None:
        """Visualiza la distribución de fallas."""
        failure_counts = df['failure'].value_counts()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        colors = ['#2ecc71', '#e74c3c']
        axes[0].bar(['No Falla (0)', 'Falla (1)'], failure_counts.values, color=colors)
        axes[0].set_title('Distribución de Fallas', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Cantidad')
        for i, v in enumerate(failure_counts.values):
            axes[0].text(i, v + 500, f'{v:,}', ha='center', fontsize=10)
        
        axes[1].pie(failure_counts.values, labels=['No Falla', 'Falla'], autopct='%1.2f%%',
                    colors=colors, explode=[0, 0.1], startangle=90)
        axes[1].set_title('Proporción de Fallas', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_model_comparison(
        results: Dict[str, Any],
        baseline_cost: float,
        y_test: np.ndarray
    ) -> None:
        """Visualiza comparación de modelos."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Métricas de clasificación
        metrics = ['AUC-ROC', 'Precision', 'Recall', 'F1-Score']
        x = np.arange(len(results))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            metric_key = metric.lower().replace('-', '_').replace(' ', '_')
            if metric == 'AUC-ROC':
                metric_key = 'auc'
            elif metric == 'F1-Score':
                metric_key = 'f1'
            values = [results[m][metric_key] for m in results]
            axes[0, 0].bar(x + i*width, values, width, label=metric)
        
        axes[0, 0].set_xlabel('Modelo')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Métricas de Clasificación por Modelo', fontweight='bold')
        axes[0, 0].set_xticks(x + width * 1.5)
        axes[0, 0].set_xticklabels(results.keys(), rotation=45, ha='right')
        axes[0, 0].legend()
        
        # Costos
        costs = [results[m]['total_cost'] for m in results]
        colors = ['#2ecc71' if c == min(costs) else '#3498db' for c in costs]
        axes[0, 1].bar(results.keys(), costs, color=colors)
        axes[0, 1].axhline(y=baseline_cost, color='r', linestyle='--',
                          label=f'Baseline: {baseline_cost:,.0f}')
        axes[0, 1].set_xlabel('Modelo')
        axes[0, 1].set_ylabel('Costo Total')
        axes[0, 1].set_title('Costo Total por Modelo', fontweight='bold')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].legend()
        
        # ROC Curves
        for name in results:
            fpr, tpr, _ = roc_curve(y_test, results[name]['y_pred_proba'])
            axes[1, 0].plot(fpr, tpr, label=f"{name} (AUC = {results[name]['auc']:.3f})")
        axes[1, 0].plot([0, 1], [0, 1], 'k--')
        axes[1, 0].set_xlabel('False Positive Rate')
        axes[1, 0].set_ylabel('True Positive Rate')
        axes[1, 0].set_title('Curvas ROC', fontweight='bold')
        axes[1, 0].legend(loc='lower right')
        
        # Precision-Recall Curves
        for name in results:
            prec, rec, _ = precision_recall_curve(y_test, results[name]['y_pred_proba'])
            axes[1, 1].plot(rec, prec, label=f"{name} (AP = {results[name]['avg_precision']:.3f})")
        axes[1, 1].set_xlabel('Recall')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].set_title('Curvas Precision-Recall', fontweight='bold')
        axes[1, 1].legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_threshold_optimization(
        threshold_df: pd.DataFrame,
        optimal_threshold: float,
        baseline_cost: float
    ) -> None:
        """Visualiza optimización de umbral."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Encontrar umbrales óptimos por diferentes criterios
        optimal_cost_idx = threshold_df['cost'].idxmin()
        optimal_cost_threshold = threshold_df.loc[optimal_cost_idx, 'threshold']
        optimal_cost_value = threshold_df.loc[optimal_cost_idx, 'cost']
        
        optimal_f1_idx = threshold_df['f1'].idxmax()
        optimal_f1_threshold = threshold_df.loc[optimal_f1_idx, 'threshold']
        
        # Costo en el umbral proporcionado (puede ser por costo o por otra métrica)
        optimal_cost = threshold_df.loc[threshold_df['threshold'] == optimal_threshold, 'cost'].values[0]
        
        # Gráfico 1: Costo vs Umbral
        axes[0].plot(threshold_df['threshold'], threshold_df['cost'], 'b-', linewidth=2)
        axes[0].axvline(x=optimal_cost_threshold, color='g', linestyle='--',
                       label=f'Óptimo costo: {optimal_cost_threshold:.2f} (${optimal_cost_value:,.0f})')
        axes[0].axvline(x=optimal_f1_threshold, color='purple', linestyle='--', alpha=0.7,
                       label=f'Óptimo F1: {optimal_f1_threshold:.2f}')
        axes[0].axvline(x=0.5, color='r', linestyle='--', alpha=0.5, label='Umbral default: 0.5')
        axes[0].axhline(y=baseline_cost, color='orange', linestyle=':',
                       label=f'Baseline: ${baseline_cost:,.0f}')
        axes[0].scatter([optimal_cost_threshold], [optimal_cost_value], color='g', s=100, zorder=5)
        axes[0].set_xlabel('Umbral de Decisión')
        axes[0].set_ylabel('Costo Total ($)')
        axes[0].set_title('Costo Total vs Umbral de Decisión', fontweight='bold')
        axes[0].legend(loc='upper right', fontsize=9)
        axes[0].grid(True, alpha=0.3)
        
        # Gráfico 2: Métricas vs Umbral (con líneas de umbral óptimo por costo y F1)
        axes[1].plot(threshold_df['threshold'], threshold_df['precision'], label='Precision', linewidth=2)
        axes[1].plot(threshold_df['threshold'], threshold_df['recall'], label='Recall', linewidth=2)
        axes[1].plot(threshold_df['threshold'], threshold_df['f1'], label='F1-Score', linewidth=2)
        axes[1].axvline(x=optimal_cost_threshold, color='g', linestyle='--',
                       label=f'Óptimo costo: {optimal_cost_threshold:.2f}')
        axes[1].axvline(x=optimal_f1_threshold, color='purple', linestyle='--', alpha=0.7,
                       label=f'Óptimo F1: {optimal_f1_threshold:.2f}')
        axes[1].set_xlabel('Umbral de Decisión')
        axes[1].set_ylabel('Score')
        axes[1].set_title('Métricas vs Umbral de Decisión', fontweight='bold')
        axes[1].legend(loc='upper right', fontsize=9)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Imprimir comparación de umbrales
        cost_at_f1 = threshold_df.loc[optimal_f1_idx, 'cost']
        f1_at_cost = threshold_df.loc[optimal_cost_idx, 'f1']
        print(f"\n📊 Comparación de Umbrales Óptimos:")
        print(f"   {'Criterio':<15} {'Umbral':<10} {'Costo':<15} {'F1':<10}")
        print(f"   {'-'*50}")
        print(f"   {'Por Costo':<15} {optimal_cost_threshold:<10.2f} ${optimal_cost_value:<14,.0f} {f1_at_cost:<10.4f}")
        print(f"   {'Por F1':<15} {optimal_f1_threshold:<10.2f} ${cost_at_f1:<14,.0f} {threshold_df.loc[optimal_f1_idx, 'f1']:<10.4f}")
    
    @staticmethod
    def plot_confusion_matrix_with_costs(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> None:
        """Visualiza matriz de confusión con costos."""
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues',
                   xticklabels=['No Falla', 'Falla'],
                   yticklabels=['No Falla', 'Falla'],
                   ax=axes[0])
        axes[0].set_xlabel('Predicción')
        axes[0].set_ylabel('Real')
        axes[0].set_title('Matriz de Confusión', fontweight='bold')
        
        cost_matrix = np.array([
            [0, fp * COST_MAINTENANCE],
            [fn * COST_FAILURE, tp * COST_MAINTENANCE]
        ])
        
        sns.heatmap(cost_matrix, annot=True, fmt=',.0f', cmap='Reds',
                   xticklabels=['No Falla', 'Falla'],
                   yticklabels=['No Falla', 'Falla'],
                   ax=axes[1])
        axes[1].set_xlabel('Predicción')
        axes[1].set_ylabel('Real')
        axes[1].set_title('Matriz de Costos ($)', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_feature_importance(
        model: Any,
        feature_cols: List[str],
        model_name: str,
        top_n: int = 20
    ) -> None:
        """Visualiza importancia de features."""
        if not hasattr(model, 'feature_importances_'):
            print("El modelo no soporta feature_importances_")
            return
        
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 10))
        
        top_features = importance_df.head(top_n)
        axes[0].barh(top_features['feature'], top_features['importance'], color='#27ae60')
        axes[0].set_xlabel('Importancia')
        axes[0].set_title(f'TOP {top_n} Features MÁS IMPORTANTES - {model_name}', fontweight='bold')
        axes[0].invert_yaxis()
        
        bottom_features = importance_df.tail(top_n)
        axes[1].barh(bottom_features['feature'], bottom_features['importance'], color='#e74c3c')
        axes[1].set_xlabel('Importancia')
        axes[1].set_title(f'TOP {top_n} Features MENOS IMPORTANTES - {model_name}', fontweight='bold')
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_roc_curve_detailed(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str = "Modelo",
        baseline_cost: Optional[float] = None
    ) -> None:
        """
        Visualiza la curva ROC detallada con puntos de umbral y área sombreada.
        Si se proporciona baseline_cost, también muestra información de costos.
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        # Encontrar el punto óptimo por Youden's J statistic
        j_scores = tpr - fpr
        optimal_j_idx = np.argmax(j_scores)
        optimal_j_threshold = thresholds[optimal_j_idx]
        
        # Si se proporciona baseline_cost, encontrar también umbral óptimo por costo
        optimal_cost_threshold = None
        optimal_cost_value = None
        if baseline_cost is not None:
            costs = []
            for thresh in thresholds:
                y_pred = (y_pred_proba >= thresh).astype(int)
                cost, _ = calculate_cost(y_true, y_pred)
                costs.append(cost)
            optimal_cost_idx = np.argmin(costs)
            optimal_cost_threshold = thresholds[optimal_cost_idx]
            optimal_cost_value = costs[optimal_cost_idx]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # ROC Curve con área sombreada
        axes[0].fill_between(fpr, tpr, alpha=0.3, color='#3498db')
        axes[0].plot(fpr, tpr, 'b-', linewidth=2, label=f'{model_name} (AUC = {auc_score:.4f})')
        axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5)')
        axes[0].scatter([fpr[optimal_j_idx]], [tpr[optimal_j_idx]], 
                       color='purple', s=100, zorder=5, 
                       label=f'Óptimo Youden (thresh={optimal_j_threshold:.2f})')
        
        # Si hay umbral óptimo por costo, marcarlo también
        if optimal_cost_threshold is not None:
            # Encontrar el índice más cercano en thresholds para el umbral óptimo por costo
            cost_idx_in_roc = np.argmin(np.abs(thresholds - optimal_cost_threshold))
            axes[0].scatter([fpr[cost_idx_in_roc]], [tpr[cost_idx_in_roc]], 
                           color='#27ae60', s=100, zorder=5, marker='s',
                           label=f'Óptimo Costo (thresh={optimal_cost_threshold:.2f})')
        
        axes[0].set_xlabel('False Positive Rate (1 - Especificidad)', fontsize=11)
        axes[0].set_ylabel('True Positive Rate (Sensibilidad)', fontsize=11)
        axes[0].set_title('Curva ROC', fontsize=12, fontweight='bold')
        axes[0].legend(loc='lower right', fontsize=9)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim([-0.01, 1.01])
        axes[0].set_ylim([-0.01, 1.01])
        
        # TPR y FPR vs Threshold
        tpr_plot = tpr[:-1] if len(tpr) > len(thresholds) else tpr
        fpr_plot = fpr[:-1] if len(fpr) > len(thresholds) else fpr
        axes[1].plot(thresholds, tpr_plot, 'g-', linewidth=2, label='TPR (Sensibilidad)')
        axes[1].plot(thresholds, fpr_plot, 'r-', linewidth=2, label='FPR (1-Especificidad)')
        axes[1].axvline(x=optimal_j_threshold, color='purple', linestyle='--', 
                       label=f'Óptimo Youden: {optimal_j_threshold:.2f}')
        if optimal_cost_threshold is not None:
            axes[1].axvline(x=optimal_cost_threshold, color='#27ae60', linestyle='--', 
                           label=f'Óptimo Costo: {optimal_cost_threshold:.2f}')
        axes[1].set_xlabel('Umbral de Decisión', fontsize=11)
        axes[1].set_ylabel('Tasa', fontsize=11)
        axes[1].set_title('TPR y FPR vs Umbral', fontsize=12, fontweight='bold')
        axes[1].legend(loc='center right', fontsize=9)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim([0, 1])
        
        plt.tight_layout()
        plt.show()
        
        # Calcular métricas en el umbral Youden
        print(f"\n📊 Métricas en umbral óptimo Youden ({optimal_j_threshold:.3f}):")
        print(f"   - TPR (Sensibilidad): {tpr[optimal_j_idx]:.4f}")
        print(f"   - FPR: {fpr[optimal_j_idx]:.4f}")
        print(f"   - Especificidad: {1 - fpr[optimal_j_idx]:.4f}")
        
        if baseline_cost is not None:
            # Calcular costo en umbral Youden
            y_pred_j = (y_pred_proba >= optimal_j_threshold).astype(int)
            cost_j, _ = calculate_cost(y_true, y_pred_j)
            
            print(f"   - Costo: ${cost_j:,.0f}")
            
            print(f"\n📊 Métricas en umbral óptimo por Costo ({optimal_cost_threshold:.3f}):")
            y_pred_cost = (y_pred_proba >= optimal_cost_threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_cost).ravel()
            tpr_cost = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr_cost = fp / (fp + tn) if (fp + tn) > 0 else 0
            print(f"   - TPR (Sensibilidad): {tpr_cost:.4f}")
            print(f"   - FPR: {fpr_cost:.4f}")
            print(f"   - Especificidad: {1 - fpr_cost:.4f}")
            print(f"   - Costo: ${optimal_cost_value:,.0f}")
            print(f"\n   💡 Diferencia de costo: ${cost_j - optimal_cost_value:,.0f}")
    
    @staticmethod
    def plot_class_separation(
        y_train: np.ndarray,
        y_test: np.ndarray,
        y_train_proba: np.ndarray,
        y_test_proba: np.ndarray
    ) -> None:
        """
        Visualiza cómo se separan las clases en train y test basado en probabilidades.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Distribución de probabilidades en TRAIN
        train_class_0 = y_train_proba[y_train == 0]
        train_class_1 = y_train_proba[y_train == 1]
        
        axes[0, 0].hist(train_class_0, bins=50, alpha=0.7, color='#2ecc71', 
                       label=f'No Falla (n={len(train_class_0):,})', density=True)
        axes[0, 0].hist(train_class_1, bins=50, alpha=0.7, color='#e74c3c', 
                       label=f'Falla (n={len(train_class_1):,})', density=True)
        axes[0, 0].axvline(x=0.5, color='black', linestyle='--', linewidth=1.5, label='Umbral 0.5')
        axes[0, 0].set_xlabel('Probabilidad de Falla')
        axes[0, 0].set_ylabel('Densidad')
        axes[0, 0].set_title('Distribución de Probabilidades - TRAIN', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Distribución de probabilidades en TEST
        test_class_0 = y_test_proba[y_test == 0]
        test_class_1 = y_test_proba[y_test == 1]
        
        axes[0, 1].hist(test_class_0, bins=50, alpha=0.7, color='#2ecc71', 
                       label=f'No Falla (n={len(test_class_0):,})', density=True)
        axes[0, 1].hist(test_class_1, bins=50, alpha=0.7, color='#e74c3c', 
                       label=f'Falla (n={len(test_class_1):,})', density=True)
        axes[0, 1].axvline(x=0.5, color='black', linestyle='--', linewidth=1.5, label='Umbral 0.5')
        axes[0, 1].set_xlabel('Probabilidad de Falla')
        axes[0, 1].set_ylabel('Densidad')
        axes[0, 1].set_title('Distribución de Probabilidades - TEST', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # KDE plots para mejor visualización
        from scipy import stats
        x_range = np.linspace(0, 1, 200)
        
        if len(train_class_0) > 1 and len(train_class_1) > 1:
            kde_train_0 = stats.gaussian_kde(train_class_0)
            kde_train_1 = stats.gaussian_kde(train_class_1)
            axes[1, 0].fill_between(x_range, kde_train_0(x_range), alpha=0.5, color='#2ecc71', label='No Falla')
            axes[1, 0].fill_between(x_range, kde_train_1(x_range), alpha=0.5, color='#e74c3c', label='Falla')
            axes[1, 0].plot(x_range, kde_train_0(x_range), color='#27ae60', linewidth=2)
            axes[1, 0].plot(x_range, kde_train_1(x_range), color='#c0392b', linewidth=2)
        axes[1, 0].axvline(x=0.5, color='black', linestyle='--', linewidth=1.5)
        axes[1, 0].set_xlabel('Probabilidad de Falla')
        axes[1, 0].set_ylabel('Densidad')
        axes[1, 0].set_title('Separación de Clases (KDE) - TRAIN', fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        if len(test_class_0) > 1 and len(test_class_1) > 1:
            kde_test_0 = stats.gaussian_kde(test_class_0)
            kde_test_1 = stats.gaussian_kde(test_class_1)
            axes[1, 1].fill_between(x_range, kde_test_0(x_range), alpha=0.5, color='#2ecc71', label='No Falla')
            axes[1, 1].fill_between(x_range, kde_test_1(x_range), alpha=0.5, color='#e74c3c', label='Falla')
            axes[1, 1].plot(x_range, kde_test_0(x_range), color='#27ae60', linewidth=2)
            axes[1, 1].plot(x_range, kde_test_1(x_range), color='#c0392b', linewidth=2)
        axes[1, 1].axvline(x=0.5, color='black', linestyle='--', linewidth=1.5)
        axes[1, 1].set_xlabel('Probabilidad de Falla')
        axes[1, 1].set_ylabel('Densidad')
        axes[1, 1].set_title('Separación de Clases (KDE) - TEST', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Estadísticas de separación
        print("\nEstadísticas de Separación de Clases:")
        print("-" * 50)
        print(f"TRAIN - No Falla: media={train_class_0.mean():.4f}, std={train_class_0.std():.4f}")
        print(f"TRAIN - Falla:    media={train_class_1.mean():.4f}, std={train_class_1.std():.4f}")
        print(f"TEST  - No Falla: media={test_class_0.mean():.4f}, std={test_class_0.std():.4f}")
        print(f"TEST  - Falla:    media={test_class_1.mean():.4f}, std={test_class_1.std():.4f}")
    
    @staticmethod
    def plot_metrics_vs_threshold(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        baseline_cost: float
    ) -> None:
        """
        Visualiza todas las métricas vs umbral de decisión.
        """
        thresholds = np.arange(0.05, 0.95, 0.02)
        
        metrics = {
            'precision': [], 'recall': [], 'f1': [], 
            'accuracy': [], 'balanced_accuracy': [],
            'specificity': [], 'cost': [], 'savings_pct': []
        }
        
        for thresh in thresholds:
            y_pred = (y_pred_proba >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            metrics['precision'].append(precision_score(y_true, y_pred, zero_division=0))
            metrics['recall'].append(recall_score(y_true, y_pred, zero_division=0))
            metrics['f1'].append(f1_score(y_true, y_pred, zero_division=0))
            metrics['accuracy'].append(accuracy_score(y_true, y_pred))
            metrics['balanced_accuracy'].append(balanced_accuracy_score(y_true, y_pred))
            metrics['specificity'].append(tn / (tn + fp) if (tn + fp) > 0 else 0)
            
            cost, _ = calculate_cost(y_true, y_pred)
            metrics['cost'].append(cost)
            metrics['savings_pct'].append((baseline_cost - cost) / baseline_cost * 100)
        
        # Encontrar umbrales óptimos
        optimal_cost_idx = np.argmin(metrics['cost'])
        optimal_cost_thresh = thresholds[optimal_cost_idx]
        optimal_cost_value = metrics['cost'][optimal_cost_idx]
        
        optimal_f1_idx = np.argmax(metrics['f1'])
        optimal_f1_thresh = thresholds[optimal_f1_idx]
        optimal_f1_value = metrics['f1'][optimal_f1_idx]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Precision, Recall, F1 con umbrales óptimos
        axes[0, 0].plot(thresholds, metrics['precision'], 'b-', linewidth=2, label='Precision')
        axes[0, 0].plot(thresholds, metrics['recall'], 'g-', linewidth=2, label='Recall')
        axes[0, 0].plot(thresholds, metrics['f1'], 'r-', linewidth=2, label='F1-Score')
        axes[0, 0].axvline(x=optimal_cost_thresh, color='#27ae60', linestyle='--', linewidth=2,
                          label=f'Óptimo Costo: {optimal_cost_thresh:.2f}')
        axes[0, 0].axvline(x=optimal_f1_thresh, color='purple', linestyle='--', linewidth=2, alpha=0.7,
                          label=f'Óptimo F1: {optimal_f1_thresh:.2f}')
        axes[0, 0].axvline(x=0.5, color='gray', linestyle=':', alpha=0.5, label='Default: 0.5')
        axes[0, 0].set_xlabel('Umbral')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Precision, Recall y F1 vs Umbral', fontweight='bold')
        axes[0, 0].legend(loc='upper right', fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xlim([0, 1])
        axes[0, 0].set_ylim([0, 1])
        
        # NUEVO: Costo Total vs Umbral (reemplaza Accuracy)
        axes[0, 1].plot(thresholds, metrics['cost'], 'b-', linewidth=2)
        axes[0, 1].fill_between(thresholds, metrics['cost'], alpha=0.2, color='blue')
        axes[0, 1].axvline(x=optimal_cost_thresh, color='#27ae60', linestyle='--', linewidth=2,
                          label=f'Óptimo Costo: {optimal_cost_thresh:.2f}')
        axes[0, 1].axvline(x=optimal_f1_thresh, color='purple', linestyle='--', linewidth=2, alpha=0.7,
                          label=f'Óptimo F1: {optimal_f1_thresh:.2f}')
        axes[0, 1].axhline(y=baseline_cost, color='red', linestyle=':', linewidth=2,
                          label=f'Baseline: ${baseline_cost:,.0f}')
        axes[0, 1].scatter([optimal_cost_thresh], [optimal_cost_value], color='#27ae60', s=100, zorder=5,
                          label=f'Mín: ${optimal_cost_value:,.0f}')
        axes[0, 1].set_xlabel('Umbral')
        axes[0, 1].set_ylabel('Costo Total ($)')
        axes[0, 1].set_title('Costo Total vs Umbral', fontweight='bold')
        axes[0, 1].legend(loc='upper right', fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xlim([0, 1])
        
        # Recall vs Specificity (Trade-off) con umbrales óptimos
        axes[1, 0].plot(thresholds, metrics['recall'], 'g-', linewidth=2, label='Recall (Sensibilidad)')
        axes[1, 0].plot(thresholds, metrics['specificity'], 'orange', linewidth=2, label='Especificidad')
        axes[1, 0].fill_between(thresholds, metrics['recall'], metrics['specificity'], 
                               alpha=0.2, color='gray')
        axes[1, 0].axvline(x=optimal_cost_thresh, color='#27ae60', linestyle='--', linewidth=2,
                          label=f'Óptimo Costo: {optimal_cost_thresh:.2f}')
        axes[1, 0].axvline(x=optimal_f1_thresh, color='purple', linestyle='--', linewidth=2, alpha=0.7,
                          label=f'Óptimo F1: {optimal_f1_thresh:.2f}')
        axes[1, 0].set_xlabel('Umbral')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Trade-off: Recall vs Especificidad', fontweight='bold')
        axes[1, 0].legend(loc='center right', fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlim([0, 1])
        axes[1, 0].set_ylim([0, 1])
        
        # Ahorro (%) vs Umbral con comparación
        axes[1, 1].plot(thresholds, metrics['savings_pct'], 'b-', linewidth=2)
        axes[1, 1].fill_between(thresholds, metrics['savings_pct'], alpha=0.3, color='blue')
        axes[1, 1].axvline(x=optimal_cost_thresh, color='#27ae60', linestyle='--', linewidth=2, 
                          label=f'Óptimo Costo: {optimal_cost_thresh:.2f}')
        axes[1, 1].axvline(x=optimal_f1_thresh, color='purple', linestyle='--', linewidth=2, alpha=0.7,
                          label=f'Óptimo F1: {optimal_f1_thresh:.2f}')
        axes[1, 1].scatter([optimal_cost_thresh], [metrics['savings_pct'][optimal_cost_idx]], 
                          color='#27ae60', s=100, zorder=5)
        axes[1, 1].axhline(y=0, color='red', linestyle=':', alpha=0.5, label='Sin ahorro')
        axes[1, 1].set_xlabel('Umbral')
        axes[1, 1].set_ylabel('Ahorro (%)')
        axes[1, 1].set_title('Ahorro de Costos vs Umbral', fontweight='bold')
        axes[1, 1].legend(loc='upper right', fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlim([0, 1])
        
        plt.tight_layout()
        plt.show()
        
        # Imprimir resumen comparativo
        cost_at_f1 = metrics['cost'][optimal_f1_idx]
        f1_at_cost = metrics['f1'][optimal_cost_idx]
        recall_at_cost = metrics['recall'][optimal_cost_idx]
        recall_at_f1 = metrics['recall'][optimal_f1_idx]
        
        print(f"\n📊 Comparación de Umbrales Óptimos:")
        print(f"   {'Criterio':<12} {'Umbral':<8} {'Costo':<12} {'Ahorro %':<10} {'F1':<8} {'Recall':<8}")
        print(f"   {'-'*58}")
        print(f"   {'Por Costo':<12} {optimal_cost_thresh:<8.2f} ${optimal_cost_value:<11,.0f} {metrics['savings_pct'][optimal_cost_idx]:<10.1f} {f1_at_cost:<8.4f} {recall_at_cost:<8.4f}")
        print(f"   {'Por F1':<12} {optimal_f1_thresh:<8.2f} ${cost_at_f1:<11,.0f} {metrics['savings_pct'][optimal_f1_idx]:<10.1f} {optimal_f1_value:<8.4f} {recall_at_f1:<8.4f}")
        print(f"\n   💡 Diferencia de costo: ${cost_at_f1 - optimal_cost_value:,.0f} (optimizar por costo ahorra más)")
    
    @staticmethod
    def plot_learning_curves(
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str = "Modelo",
        cv: int = 5,
        n_jobs: int = -1,
        train_sizes: np.ndarray = np.linspace(0.1, 1.0, 10)
    ) -> None:
        """
        Visualiza las curvas de aprendizaje del modelo.
        """
        print(f"Calculando curvas de aprendizaje para {model_name}...")
        
        train_sizes_abs, train_scores, test_scores = learning_curve(
            model, X, y, cv=cv, n_jobs=n_jobs,
            train_sizes=train_sizes, scoring='roc_auc',
            shuffle=True, random_state=42
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Learning Curve
        axes[0].fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, 
                            alpha=0.2, color='blue')
        axes[0].fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std, 
                            alpha=0.2, color='orange')
        axes[0].plot(train_sizes_abs, train_mean, 'b-', linewidth=2, marker='o', label='Train Score')
        axes[0].plot(train_sizes_abs, test_mean, 'orange', linewidth=2, marker='s', label='Validation Score')
        axes[0].set_xlabel('Tamaño del Conjunto de Entrenamiento')
        axes[0].set_ylabel('AUC-ROC Score')
        axes[0].set_title(f'Curva de Aprendizaje - {model_name}', fontweight='bold')
        axes[0].legend(loc='lower right')
        axes[0].grid(True, alpha=0.3)
        
        # Gap Analysis
        gap = train_mean - test_mean
        axes[1].bar(range(len(train_sizes_abs)), gap, color=['green' if g < 0.05 else 'orange' if g < 0.1 else 'red' for g in gap])
        axes[1].axhline(y=0.05, color='green', linestyle='--', label='Gap óptimo (< 0.05)')
        axes[1].axhline(y=0.1, color='orange', linestyle='--', label='Gap aceptable (< 0.1)')
        axes[1].set_xlabel('Índice de Tamaño de Entrenamiento')
        axes[1].set_ylabel('Gap (Train - Validation)')
        axes[1].set_title('Análisis de Overfitting (Gap)', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nResumen de Curvas de Aprendizaje:")
        print(f"  - Score final en Train: {train_mean[-1]:.4f} (+/- {train_std[-1]:.4f})")
        print(f"  - Score final en Validation: {test_mean[-1]:.4f} (+/- {test_std[-1]:.4f})")
        print(f"  - Gap final: {gap[-1]:.4f}")
        if gap[-1] < 0.05:
            print("  - Estado: Buen ajuste (Low variance)")
        elif gap[-1] < 0.1:
            print("  - Estado: Ajuste aceptable (Moderate variance)")
        else:
            print("  - Estado: Posible overfitting (High variance)")
    
    @staticmethod
    def plot_calibration_curve(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str = "Modelo",
        n_bins: int = 10
    ) -> None:
        """
        Visualiza la curva de calibración del modelo.
        """
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins, strategy='uniform')
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Calibration curve
        axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfectamente Calibrado')
        axes[0].plot(prob_pred, prob_true, 's-', color='#3498db', linewidth=2, 
                    markersize=8, label=model_name)
        axes[0].fill_between(prob_pred, prob_true, prob_pred, alpha=0.2, color='blue')
        axes[0].set_xlabel('Probabilidad Predicha Media')
        axes[0].set_ylabel('Proporción de Positivos Real')
        axes[0].set_title('Curva de Calibración', fontweight='bold')
        axes[0].legend(loc='lower right')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim([0, 1])
        axes[0].set_ylim([0, 1])
        
        # Histogram of predicted probabilities
        axes[1].hist(y_pred_proba[y_true == 0], bins=50, alpha=0.7, color='#2ecc71', 
                    label='No Falla', density=True)
        axes[1].hist(y_pred_proba[y_true == 1], bins=50, alpha=0.7, color='#e74c3c', 
                    label='Falla', density=True)
        axes[1].set_xlabel('Probabilidad Predicha')
        axes[1].set_ylabel('Densidad')
        axes[1].set_title('Distribución de Probabilidades por Clase', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Brier score
        from sklearn.metrics import brier_score_loss
        brier = brier_score_loss(y_true, y_pred_proba)
        print(f"\nMétricas de Calibración:")
        print(f"  - Brier Score: {brier:.4f} (menor es mejor, 0 = perfecto)")
        print(f"  - Interpretación: {'Bien calibrado' if brier < 0.1 else 'Calibración moderada' if brier < 0.25 else 'Mal calibrado'}")
    
    @staticmethod
    def plot_train_test_comparison(
        y_train: np.ndarray,
        y_test: np.ndarray,
        y_train_pred: np.ndarray,
        y_test_pred: np.ndarray,
        y_train_proba: np.ndarray,
        y_test_proba: np.ndarray
    ) -> None:
        """
        Compara el rendimiento del modelo en train vs test.
        
        Análisis de Overfitting:
        - Gap = Train - Test para cada métrica
        - Gap positivo grande → Overfitting (modelo memoriza train)
        - Gap cercano a 0 → Buena generalización
        - Gap > 0.05 → Señal de overfitting preocupante
        """
        # Calcular costos
        train_cost, _ = calculate_cost(y_train, y_train_pred)
        test_cost, _ = calculate_cost(y_test, y_test_pred)
        train_baseline = calculate_baseline_cost(y_train)
        test_baseline = calculate_baseline_cost(y_test)
        
        # Métricas en train (normalizadas 0-1)
        train_metrics = {
            'AUC-ROC': roc_auc_score(y_train, y_train_proba),
            'Precision': precision_score(y_train, y_train_pred),
            'Recall': recall_score(y_train, y_train_pred),
            'F1-Score': f1_score(y_train, y_train_pred)
        }
        
        # Métricas en test (normalizadas 0-1)
        test_metrics = {
            'AUC-ROC': roc_auc_score(y_test, y_test_proba),
            'Precision': precision_score(y_test, y_test_pred),
            'Recall': recall_score(y_test, y_test_pred),
            'F1-Score': f1_score(y_test, y_test_pred)
        }
        
        # Ahorro relativo (para comparar en misma escala)
        train_savings_pct = (train_baseline - train_cost) / train_baseline * 100
        test_savings_pct = (test_baseline - test_cost) / test_baseline * 100
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # ===== Gráfico 1: Comparación de métricas =====
        x = np.arange(len(train_metrics))
        width = 0.35
        
        train_values = list(train_metrics.values())
        test_values = list(test_metrics.values())
        
        bars1 = axes[0, 0].bar(x - width/2, train_values, width, label='Train', color='#3498db')
        bars2 = axes[0, 0].bar(x + width/2, test_values, width, label='Test', color='#e74c3c')
        
        axes[0, 0].set_xlabel('Métrica')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Comparación de Métricas: Train vs Test', fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(train_metrics.keys(), rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        axes[0, 0].set_ylim([0, 1])
        
        for bar, val in zip(bars1, train_values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                          f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        for bar, val in zip(bars2, test_values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                          f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        # ===== Gráfico 2: Comparación de COSTOS =====
        cost_labels = ['Costo\nModelo', 'Baseline\n(sin modelo)', 'Ahorro\n($)']
        train_cost_values = [train_cost, train_baseline, train_baseline - train_cost]
        test_cost_values = [test_cost, test_baseline, test_baseline - test_cost]
        
        x_cost = np.arange(len(cost_labels))
        bars_cost1 = axes[0, 1].bar(x_cost - width/2, train_cost_values, width, label='Train', color='#3498db')
        bars_cost2 = axes[0, 1].bar(x_cost + width/2, test_cost_values, width, label='Test', color='#e74c3c')
        
        axes[0, 1].set_ylabel('Costo ($)')
        axes[0, 1].set_title('Comparación de COSTOS: Train vs Test', fontweight='bold')
        axes[0, 1].set_xticks(x_cost)
        axes[0, 1].set_xticklabels(cost_labels)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars_cost1, train_cost_values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                          f'${val:,.0f}', ha='center', va='bottom', fontsize=9)
        for bar, val in zip(bars_cost2, test_cost_values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                          f'${val:,.0f}', ha='center', va='bottom', fontsize=9)
        
        # ===== Gráfico 3: Gap de métricas (Overfitting) =====
        gaps = [train_metrics[k] - test_metrics[k] for k in train_metrics.keys()]
        colors = ['#27ae60' if abs(g) < 0.02 else '#f39c12' if abs(g) < 0.05 else '#e74c3c' for g in gaps]
        axes[1, 0].bar(train_metrics.keys(), gaps, color=colors)
        axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[1, 0].axhline(y=0.02, color='#27ae60', linestyle='--', alpha=0.7, label='Gap ideal (< 0.02)')
        axes[1, 0].axhline(y=-0.02, color='#27ae60', linestyle='--', alpha=0.7)
        axes[1, 0].axhline(y=0.05, color='#f39c12', linestyle='--', alpha=0.7, label='Gap aceptable (< 0.05)')
        axes[1, 0].axhline(y=-0.05, color='#f39c12', linestyle='--', alpha=0.7)
        axes[1, 0].set_xlabel('Métrica')
        axes[1, 0].set_ylabel('Gap (Train - Test)')
        axes[1, 0].set_title('Análisis de Overfitting (Gap de Métricas)', fontweight='bold')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].legend(loc='upper right', fontsize=8)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # ===== Gráfico 4: Comparación de Ahorro % =====
        savings_labels = ['Ahorro (%)']
        x_sav = np.arange(1)
        bars_sav1 = axes[1, 1].bar(x_sav - width/2, [train_savings_pct], width, label='Train', color='#3498db')
        bars_sav2 = axes[1, 1].bar(x_sav + width/2, [test_savings_pct], width, label='Test', color='#e74c3c')
        
        # Gap de ahorro
        savings_gap = train_savings_pct - test_savings_pct
        gap_color = '#27ae60' if abs(savings_gap) < 5 else '#f39c12' if abs(savings_gap) < 10 else '#e74c3c'
        
        axes[1, 1].set_ylabel('Ahorro (%)')
        axes[1, 1].set_title(f'Ahorro vs Baseline (Gap: {savings_gap:.1f}%)', fontweight='bold')
        axes[1, 1].set_xticks(x_sav)
        axes[1, 1].set_xticklabels(savings_labels)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        axes[1, 1].axhline(y=0, color='red', linestyle=':', alpha=0.5, label='Sin ahorro')
        
        for bar, val in zip(bars_sav1, [train_savings_pct]):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                          f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        for bar, val in zip(bars_sav2, [test_savings_pct]):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                          f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # ===== Resumen impreso =====
        print("\n" + "="*70)
        print("📊 ANÁLISIS DE GENERALIZACIÓN (OVERFITTING)")
        print("="*70)
        print("\n1️⃣  MÉTRICAS DE CLASIFICACIÓN:")
        print("-" * 60)
        print(f"   {'Métrica':<12} {'Train':>10} {'Test':>10} {'Gap':>10} {'Estado':<8}")
        print("-" * 60)
        for metric in train_metrics.keys():
            gap = train_metrics[metric] - test_metrics[metric]
            status = '✓ OK' if abs(gap) < 0.02 else '⚠ Revisar' if abs(gap) < 0.05 else '✗ Overfit'
            print(f"   {metric:<12} {train_metrics[metric]:>10.4f} {test_metrics[metric]:>10.4f} {gap:>+10.4f} {status:<8}")
        
        print("\n2️⃣  ANÁLISIS DE COSTOS:")
        print("-" * 60)
        print(f"   {'Concepto':<20} {'Train':>15} {'Test':>15}")
        print("-" * 60)
        print(f"   {'Baseline (sin modelo)':<20} ${train_baseline:>14,.0f} ${test_baseline:>14,.0f}")
        print(f"   {'Costo con modelo':<20} ${train_cost:>14,.0f} ${test_cost:>14,.0f}")
        print(f"   {'Ahorro ($)':<20} ${train_baseline-train_cost:>14,.0f} ${test_baseline-test_cost:>14,.0f}")
        print(f"   {'Ahorro (%)':<20} {train_savings_pct:>14.1f}% {test_savings_pct:>14.1f}%")
        
        print(f"\n3️⃣  DIAGNÓSTICO:")
        avg_gap = np.mean([abs(g) for g in gaps])
        if avg_gap < 0.02:
            print("   ✅ El modelo generaliza BIEN (bajo overfitting)")
        elif avg_gap < 0.05:
            print("   ⚠️  Generalización ACEPTABLE (monitorear)")
        else:
            print("   ❌ POSIBLE OVERFITTING (gap promedio > 0.05)")
        
        if abs(savings_gap) < 5:
            print(f"   ✅ Ahorro consistente entre Train y Test (gap: {savings_gap:.1f}%)")
        else:
            print(f"   ⚠️  Diferencia de ahorro significativa (gap: {savings_gap:.1f}%) - revisar generalización de costos")
    
    @staticmethod
    def plot_prediction_distribution_over_time(
        df: pd.DataFrame,
        y_pred_proba: np.ndarray,
        y_true: np.ndarray
    ) -> None:
        """
        Visualiza la distribución de predicciones a lo largo del tiempo.
        """
        df_plot = df.copy()
        df_plot['pred_proba'] = y_pred_proba
        df_plot['true_label'] = y_true
        
        # Agrupar por fecha
        daily_stats = df_plot.groupby('date').agg({
            'pred_proba': ['mean', 'std', 'min', 'max'],
            'true_label': ['sum', 'count']
        })
        daily_stats.columns = ['prob_mean', 'prob_std', 'prob_min', 'prob_max', 'failures', 'total']
        daily_stats['failure_rate'] = daily_stats['failures'] / daily_stats['total'] * 100
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # Probabilidad promedio vs tiempo
        axes[0].plot(daily_stats.index, daily_stats['prob_mean'], 'b-', linewidth=1, label='Prob. Media')
        axes[0].fill_between(daily_stats.index, 
                            daily_stats['prob_mean'] - daily_stats['prob_std'],
                            daily_stats['prob_mean'] + daily_stats['prob_std'],
                            alpha=0.3, color='blue', label='± 1 Std')
        axes[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Umbral 0.5')
        axes[0].set_xlabel('Fecha')
        axes[0].set_ylabel('Probabilidad de Falla')
        axes[0].set_title('Probabilidad de Falla Predicha a lo Largo del Tiempo', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Tasa de fallas real vs tiempo
        axes[1].bar(daily_stats.index, daily_stats['failure_rate'], alpha=0.7, color='#e74c3c')
        axes[1].plot(daily_stats.index, daily_stats['prob_mean'] * 100, 'b-', linewidth=2, 
                    label='Prob. Media (%)')
        axes[1].set_xlabel('Fecha')
        axes[1].set_ylabel('Tasa de Fallas (%)')
        axes[1].set_title('Tasa de Fallas Real vs Predicción', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_optuna_optimization_history(study: 'optuna.Study') -> None:
        """
        Visualiza el historial de optimización de Optuna.
        """
        trials_df = study.trials_dataframe()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Historia de optimización
        axes[0].plot(trials_df.index, trials_df['value'], 'b-', alpha=0.5, linewidth=1)
        axes[0].scatter(trials_df.index, trials_df['value'], c=trials_df['value'], 
                       cmap='RdYlGn', alpha=0.7, s=50)
        
        # Mejor valor acumulado
        best_values = trials_df['value'].cummax()
        axes[0].plot(trials_df.index, best_values, 'g-', linewidth=2, label='Mejor Acumulado')
        axes[0].axhline(y=study.best_value, color='green', linestyle='--', 
                       label=f'Mejor: {study.best_value:.4f}')
        axes[0].set_xlabel('Trial')
        axes[0].set_ylabel('F1-Score')
        axes[0].set_title('Historia de Optimización - Optuna', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Importancia de hiperparámetros
        try:
            from optuna.importance import get_param_importances
            param_importance = get_param_importances(study)
            params_sorted = sorted(param_importance.items(), key=lambda x: x[1], reverse=True)
            param_names = [p[0] for p in params_sorted]
            param_values = [p[1] for p in params_sorted]
            
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(param_names)))
            axes[1].barh(param_names, param_values, color=colors)
            axes[1].set_xlabel('Importancia')
            axes[1].set_title('Importancia de Hiperparámetros', fontweight='bold')
            axes[1].invert_yaxis()
            axes[1].grid(True, alpha=0.3, axis='x')
        except Exception as e:
            axes[1].text(0.5, 0.5, f'No se pudo calcular\nimportancia de parámetros\n{str(e)}', 
                        ha='center', va='center', fontsize=10, transform=axes[1].transAxes)
            axes[1].set_title('Importancia de Hiperparámetros', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nResumen de Optimización:")
        print(f"  - Total de trials: {len(trials_df)}")
        print(f"  - Mejor F1-Score: {study.best_value:.4f}")
        print(f"  - Trial del mejor resultado: {study.best_trial.number}")


# =============================================================================
# PREDICCIÓN EN NUEVOS DATOS
# =============================================================================
def predict_failure_probability(
    new_data: pd.DataFrame,
    model_path: str = 'predictive_maintenance_model_optimized.pkl'
) -> pd.DataFrame:
    """
    Predice la probabilidad de falla para nuevos datos.
    
    Args:
        new_data: DataFrame con las mismas features que el modelo
        model_path: Ruta al modelo guardado
    
    Returns:
        DataFrame con predicciones y probabilidades
    """
    artifacts = joblib.load(model_path)
    
    # Preprocesar
    X_new = new_data[artifacts['feature_cols']]
    X_new_imputed = artifacts['imputer'].transform(X_new)
    X_new_scaled = artifacts['scaler'].transform(X_new_imputed)
    
    # Predecir
    probabilities = artifacts['model'].predict_proba(X_new_scaled)[:, 1]
    predictions = (probabilities >= artifacts['optimal_threshold']).astype(int)
    
    return pd.DataFrame({
        'probability': probabilities,
        'prediction': predictions,
        'maintenance_recommended': ['Sí' if p == 1 else 'No' for p in predictions]
    })


def save_model_artifacts(
    model: Any,
    scaler: RobustScaler,
    imputer: SimpleImputer,
    optimal_threshold: float,
    feature_cols: List[str],
    model_name: str,
    best_params: Dict[str, Any],
    best_f1: float,
    output_path: str = 'predictive_maintenance_model_optimized.pkl'
) -> None:
    """Guarda los artefactos del modelo."""
    model_artifacts = {
        'model': model,
        'scaler': scaler,
        'imputer': imputer,
        'optimal_threshold': optimal_threshold,
        'feature_cols': feature_cols,
        'model_name': model_name,
        'best_params': best_params,
        'best_f1': best_f1
    }
    
    joblib.dump(model_artifacts, output_path)
    print(f"✅ Modelo guardado en '{output_path}'")


# =============================================================================
# FUNCIONES DE ANÁLISIS CON PRINTS
# =============================================================================
def analyze_dataset(df: pd.DataFrame) -> None:
    """Analiza y muestra información del dataset."""
    print_section("INFORMACIÓN GENERAL DEL DATASET")
    print(f"\nDimensiones: {df.shape[0]:,} filas x {df.shape[1]} columnas")
    print(f"\nColumnas: {df.columns.tolist()}")
    print(f"\nTipos de datos:\n{df.dtypes}")


def plot_cost_over_time(df: pd.DataFrame, date_col: str = 'date', failure_col: str = 'failure') -> None:
    """
    Grafica el costo acumulado de fallas vs mantenimiento preventivo a lo largo del tiempo.
    
    Muestra dos escenarios:
    1. No hacer nada: cada falla cuesta COST_FAILURE
    2. Predicción perfecta: mantener solo los que fallan cuesta COST_MAINTENANCE por falla
    """
    # Asegurar que date es datetime
    df_plot = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_plot[date_col]):
        df_plot[date_col] = pd.to_datetime(df_plot[date_col])
    
    # Agrupar por fecha
    daily = df_plot.groupby(date_col)[failure_col].agg(['sum', 'count']).reset_index()
    daily.columns = [date_col, 'failures', 'total']
    daily = daily.sort_values(date_col)
    
    # Calcular costos acumulados
    daily['cost_no_action'] = (daily['failures'] * COST_FAILURE).cumsum()
    daily['cost_perfect_prediction'] = (daily['failures'] * COST_MAINTENANCE).cumsum()
    
    # Totales finales
    total_failures = daily['failures'].sum()
    total_records = daily['total'].sum()
    final_cost_no_action = total_failures * COST_FAILURE
    final_cost_perfect = total_failures * COST_MAINTENANCE
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(daily[date_col], daily['cost_no_action'], 
            color='#e74c3c', linewidth=2.5, label=f'No hacer nada (${final_cost_no_action:,.0f})')
    ax.plot(daily[date_col], daily['cost_perfect_prediction'], 
            color='#27ae60', linewidth=2.5, linestyle='--', 
            label=f'Prediccion perfecta (${final_cost_perfect:,.0f})')
    
    # Área entre las curvas para mostrar ahorro potencial
    ax.fill_between(daily[date_col], daily['cost_no_action'], daily['cost_perfect_prediction'],
                    alpha=0.2, color='#27ae60', label='Ahorro potencial')
    
    ax.set_xlabel('Fecha', fontsize=12)
    ax.set_ylabel('Costo Acumulado ($)', fontsize=12)
    ax.set_title('Costo Acumulado: Escenarios de Mantenimiento', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Formato de fechas en eje x
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Resumen
    print(f"\n{'='*60}")
    print("RESUMEN DE COSTOS")
    print(f"{'='*60}")
    print(f"Total de registros: {total_records:,}")
    print(f"Total de fallas: {total_failures:,} ({total_failures/total_records*100:.2f}%)")
    print(f"\nEscenarios:")
    print(f"  1. No hacer nada:        ${final_cost_no_action:,.2f}")
    print(f"  2. Prediccion perfecta:  ${final_cost_perfect:,.2f}")
    print(f"\nAhorro maximo posible: ${final_cost_no_action - final_cost_perfect:,.2f} ({(final_cost_no_action - final_cost_perfect)/final_cost_no_action*100:.1f}%)")


def plot_cost_over_time_with_model(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    date_col: str = 'date',
    model_name: str = 'Modelo'
) -> None:
    """
    Grafica el costo acumulado comparando tres escenarios:
    1. No hacer nada: cada falla cuesta COST_FAILURE
    2. Predicción perfecta: mantener solo los que fallan cuesta COST_MAINTENANCE por falla
    3. El modelo: según sus predicciones (FN=COST_FAILURE, TP/FP=COST_MAINTENANCE)
    
    Args:
        df: DataFrame con la columna de fechas
        y_pred: Predicciones del modelo (0 o 1)
        y_true: Valores reales (0 o 1)
        date_col: Nombre de la columna de fechas
        model_name: Nombre del modelo para la leyenda
    """
    df_plot = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_plot[date_col]):
        df_plot[date_col] = pd.to_datetime(df_plot[date_col])
    
    # Agregar predicciones y valores reales al DataFrame
    df_plot = df_plot.reset_index(drop=True)
    df_plot['y_true'] = y_true
    df_plot['y_pred'] = y_pred
    
    # Calcular costo por registro para cada escenario
    # No hacer nada: costo = y_true * COST_FAILURE (si falla, pago)
    df_plot['cost_no_action'] = df_plot['y_true'] * COST_FAILURE
    
    # Predicción perfecta: costo = y_true * COST_MAINTENANCE (si falla, hago mantenimiento preventivo)
    df_plot['cost_perfect'] = df_plot['y_true'] * COST_MAINTENANCE
    
    # Modelo: si predigo 1, pago COST_MAINTENANCE; si predigo 0 y falla, pago COST_FAILURE
    df_plot['cost_model'] = np.where(
        df_plot['y_pred'] == 1,
        COST_MAINTENANCE,  # Hago mantenimiento
        df_plot['y_true'] * COST_FAILURE  # No hago nada, pago si falla
    )
    
    # Agrupar por fecha
    daily = df_plot.groupby(date_col).agg({
        'y_true': 'sum',  # Fallas reales del día
        'cost_no_action': 'sum',
        'cost_perfect': 'sum',
        'cost_model': 'sum'
    }).reset_index()
    daily.columns = [date_col, 'failures', 'cost_no_action', 'cost_perfect', 'cost_model']
    daily = daily.sort_values(date_col)
    
    # Calcular costos acumulados
    daily['cost_no_action_cum'] = daily['cost_no_action'].cumsum()
    daily['cost_perfect_cum'] = daily['cost_perfect'].cumsum()
    daily['cost_model_cum'] = daily['cost_model'].cumsum()
    
    # Totales finales
    final_cost_no_action = daily['cost_no_action'].sum()
    final_cost_perfect = daily['cost_perfect'].sum()
    final_cost_model = daily['cost_model'].sum()
    total_records = len(df_plot)
    total_failures = int(daily['failures'].sum())
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Línea de no hacer nada (peor escenario)
    ax.plot(daily[date_col], daily['cost_no_action_cum'], 
            color='#e74c3c', linewidth=2.5, 
            label=f'No hacer nada (${final_cost_no_action:,.0f})')
    
    # Línea del modelo
    ax.plot(daily[date_col], daily['cost_model_cum'], 
            color='#3498db', linewidth=2.5, 
            label=f'{model_name} (${final_cost_model:,.0f})')
    
    # Línea de predicción perfecta (mejor escenario posible)
    ax.plot(daily[date_col], daily['cost_perfect_cum'], 
            color='#27ae60', linewidth=2.5, linestyle='--', 
            label=f'Predicción perfecta (${final_cost_perfect:,.0f})')
    
    # Área de ahorro del modelo vs no hacer nada
    ax.fill_between(daily[date_col], daily['cost_no_action_cum'], daily['cost_model_cum'],
                    alpha=0.15, color='#3498db', label='Ahorro del modelo')
    
    # Área de mejora potencial (modelo vs perfecto)
    ax.fill_between(daily[date_col], daily['cost_model_cum'], daily['cost_perfect_cum'],
                    alpha=0.15, color='#f39c12', label='Margen de mejora')
    
    ax.set_xlabel('Fecha', fontsize=12)
    ax.set_ylabel('Costo Acumulado ($)', fontsize=12)
    ax.set_title('Comparación de Costos: Benchmarks vs Modelo Final', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Formato de fechas en eje x
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Calcular métricas de ahorro
    ahorro_vs_nada = final_cost_no_action - final_cost_model
    ahorro_pct_vs_nada = (ahorro_vs_nada / final_cost_no_action * 100) if final_cost_no_action > 0 else 0
    ahorro_max_posible = final_cost_no_action - final_cost_perfect
    eficiencia_modelo = (ahorro_vs_nada / ahorro_max_posible * 100) if ahorro_max_posible > 0 else 0
    
    # Resumen
    print(f"\n{'='*70}")
    print("RESUMEN DE COSTOS - COMPARACIÓN DE ESCENARIOS")
    print(f"{'='*70}")
    print(f"\nDataset de test:")
    print(f"  - Total de registros: {total_records:,}")
    print(f"  - Total de fallas: {total_failures:,} ({total_failures/total_records*100:.2f}%)")
    
    print(f"\nCostos totales:")
    print(f"  1. No hacer nada:        ${final_cost_no_action:,.2f}")
    print(f"  2. {model_name:20s}    ${final_cost_model:,.2f}")
    print(f"  3. Predicción perfecta:  ${final_cost_perfect:,.2f}")
    
    print(f"\nAnálisis de ahorro:")
    print(f"  - Ahorro del modelo vs no hacer nada: ${ahorro_vs_nada:,.2f} ({ahorro_pct_vs_nada:.1f}%)")
    print(f"  - Ahorro máximo posible:              ${ahorro_max_posible:,.2f}")
    print(f"  - Eficiencia del modelo:              {eficiencia_modelo:.1f}% del ahorro máximo")
    
    # Veredicto
    print(f"\n{'='*70}")
    if final_cost_model < final_cost_no_action:
        print(f"✅ EL MODELO VENCE al benchmark 'No hacer nada' por ${ahorro_vs_nada:,.2f}")
    else:
        print(f"❌ El modelo NO vence al benchmark 'No hacer nada'")
    
    if final_cost_model <= final_cost_perfect * 1.1:  # Dentro del 10% del óptimo
        print(f"🏆 EL MODELO está muy cerca de la predicción perfecta!")
    elif final_cost_model < (final_cost_no_action + final_cost_perfect) / 2:
        print(f"👍 El modelo está más cerca de la predicción perfecta que de no hacer nada")
    else:
        print(f"⚠️  El modelo tiene margen de mejora significativo")
    print(f"{'='*70}")


def analyze_nulls(df: pd.DataFrame) -> None:
    """Analiza valores nulos del dataset."""
    print_section("ANÁLISIS DE VALORES NULOS")
    null_counts = df.isnull().sum()
    null_pct = (df.isnull().sum() / len(df) * 100).round(2)
    null_df = pd.DataFrame({'Nulos': null_counts, 'Porcentaje (%)': null_pct})
    null_with_values = null_df[null_df['Nulos'] > 0]
    if len(null_with_values) > 0:
        print(null_with_values)
    else:
        print("No hay valores nulos en el dataset")


def analyze_target_distribution(df: pd.DataFrame, target_col: str = 'failure') -> None:
    """Analiza la distribución de la variable objetivo."""
    print_section("DISTRIBUCIÓN DE LA VARIABLE OBJETIVO")
    counts = df[target_col].value_counts()
    pct = df[target_col].value_counts(normalize=True) * 100
    print(f"\nConteo:")
    print(f"  - No falla (0): {counts[0]:,} ({pct[0]:.2f}%)")
    print(f"  - Falla (1):    {counts[1]:,} ({pct[1]:.2f}%)")
    print(f"\nRatio de desbalance: 1:{counts[0]//counts[1]}")
    print("\n⚠️  DATASET DESBALANCEADO - Se requieren técnicas de balanceo")


def analyze_temporal(df: pd.DataFrame) -> None:
    """Analiza información temporal del dataset."""
    print_section("ANÁLISIS TEMPORAL")
    print(f"\nRango de fechas: {df['date'].min()} a {df['date'].max()}")
    print(f"Duración total: {(df['date'].max() - df['date'].min()).days} días")
    print(f"Dispositivos únicos: {df['device'].nunique():,}")


def analyze_mean_comparison(df: pd.DataFrame, attributes: List[str]) -> None:
    """Compara medias de atributos entre clases."""
    print_section("COMPARACIÓN DE MEDIAS: FALLA vs NO FALLA")
    
    comparison = df.groupby('failure')[attributes].mean().T
    comparison.columns = ['No Falla (0)', 'Falla (1)']
    comparison['Diferencia (%)'] = ((comparison['Falla (1)'] - comparison['No Falla (0)']) / comparison['No Falla (0)'] * 100).round(2)
    comparison['Diferencia Absoluta'] = comparison['Falla (1)'] - comparison['No Falla (0)']
    
    print(comparison)


def analyze_correlation(df: pd.DataFrame, attributes: List[str]) -> None:
    """Analiza y visualiza la matriz de correlación."""
    print_section("MATRIZ DE CORRELACIÓN")
    
    correlation_matrix = df[attributes + ['failure']].corr()
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlBu_r', 
                center=0, fmt='.2f', square=True, linewidths=0.5,
                cbar_kws={'shrink': 0.8})
    plt.title('Matriz de Correlación', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Correlación con target
    print("\nCorrelación con 'failure':")
    correlation_with_target = correlation_matrix['failure'].drop('failure').sort_values(key=abs, ascending=False)
    print(correlation_with_target)


def analyze_device_failures(df: pd.DataFrame) -> None:
    """Analiza estadísticas de fallas por dispositivo."""
    print_section("ANÁLISIS POR DISPOSITIVO")

    device_failures = df.groupby('device').agg({
        'failure': ['sum', 'count'],
        'date': ['min', 'max']
    })
    device_failures.columns = ['total_fallas', 'dias_operacion', 'primera_fecha', 'ultima_fecha']
    device_failures['tasa_falla'] = device_failures['total_fallas'] / device_failures['dias_operacion'] * 100

    print(f"\nDispositivos con al menos una falla: {(device_failures['total_fallas'] > 0).sum()}")
    print(f"Dispositivos sin fallas: {(device_failures['total_fallas'] == 0).sum()}")
    print(f"\nTop 10 dispositivos con más fallas:")
    print(device_failures.nlargest(10, 'total_fallas')[['total_fallas', 'dias_operacion', 'tasa_falla']])


def analyze_attribute_distributions(df: pd.DataFrame, attributes: List[str]) -> None:
    """Visualiza la distribución de atributos por clase con KDE (campanas solapadas)."""
    print_section("DISTRIBUCIÓN DE ATRIBUTOS")
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    # Separar datos por clase
    df_no_failure = df[df['failure'] == 0]
    df_failure = df[df['failure'] == 1]

    for i, attr in enumerate(attributes):
        if i < len(axes):
            ax = axes[i]
            
            # Calcular medias
            mean_no_failure = df_no_failure[attr].mean()
            mean_failure = df_failure[attr].mean()
            
            # KDE plot para cada clase (distribuciones tipo campana solapadas)
            sns.kdeplot(
                data=df_no_failure[attr].dropna(), 
                ax=ax, 
                color='#2ecc71', 
                fill=True, 
                alpha=0.4, 
                label=f'No Falla (μ={mean_no_failure:.2f})',
                linewidth=2
            )
            sns.kdeplot(
                data=df_failure[attr].dropna(), 
                ax=ax, 
                color='#e74c3c', 
                fill=True, 
                alpha=0.4, 
                label=f'Falla (μ={mean_failure:.2f})',
                linewidth=2
            )
            
            # Líneas verticales para las medias
            ax.axvline(mean_no_failure, color='#27ae60', linestyle='--', linewidth=2, alpha=0.8)
            ax.axvline(mean_failure, color='#c0392b', linestyle='--', linewidth=2, alpha=0.8)
            
            ax.set_title(f'{attr}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Valor')
            ax.set_ylabel('Densidad')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.suptitle('Distribución de Atributos por Clase de Falla (KDE)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def analyze_feature_engineering_results(df: pd.DataFrame) -> None:
    """Analiza y muestra resumen compacto del feature engineering."""
    # Contar tipos de features
    rolling_cols = [c for c in df.columns if '_rolling_' in c]
    diff_cols = [c for c in df.columns if '_diff' in c or '_pct_change' in c]
    zscore_cols = [c for c in df.columns if '_zscore_' in c]
    temporal_cols = ['day_of_week', 'day_of_month', 'week_of_year', 'is_weekend', 
                     'day_of_week_sin', 'day_of_week_cos']
    temporal_cols = [c for c in temporal_cols if c in df.columns]
    device_cols = [c for c in df.columns if c.startswith('prefix_') or c == 'device_age_days' or c.startswith('device_failures') or c.startswith('device_days') or c.startswith('device_failure_rate') or c == 'device_has_failed_before']
    interaction_cols = [c for c in df.columns if 'all_attributes' in c or c in ['attr7_attr8_equal', 'attr7_is_active', 'attr8_is_active']]
    device_agg_cols = [c for c in df.columns if '_device_mean' in c or '_device_std' in c or '_device_max' in c or '_device_min' in c]
    
    null_count = df.isnull().any().sum()
    null_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
    
    print(f"""
┌─────────────────────────────────────────────────────────┐
│           FEATURE ENGINEERING - RESUMEN                 │
├─────────────────────────────────────────────────────────┤
│  Dimensiones: {df.shape[0]:,} filas × {df.shape[1]} columnas{' ' * (25 - len(f'{df.shape[0]:,}') - len(str(df.shape[1])))}│
│  Nulos: {null_count} cols con NaN ({null_pct:.1f}% del total){' ' * (26 - len(str(null_count)) - len(f'{null_pct:.1f}'))}│
├─────────────────────────────────────────────────────────┤
│  Features generadas:                                    │
│    • Rolling (mean, max):        {len(rolling_cols):>3} features           │
│    • Diferencias/Cambios:        {len(diff_cols):>3} features           │
│    • Z-Score (anomalías):        {len(zscore_cols):>3} features           │
│    • Temporales:                 {len(temporal_cols):>3} features           │
│    • Dispositivo:                {len(device_cols):>3} features           │
│    • Agregados por device:       {len(device_agg_cols):>3} features           │
│    • Interacciones:              {len(interaction_cols):>3} features           │
└─────────────────────────────────────────────────────────┘
""")


# =============================================================================
# PIPELINES DE ALTO NIVEL
# =============================================================================
def train_models_pipeline(
    model_trainer: ModelTrainer,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    baseline_cost: float
) -> Dict[str, Any]:
    """Ejecuta el pipeline de entrenamiento y evaluación."""
    print_section("ENTRENAMIENTO Y EVALUACIÓN DE MODELOS")
    
    results = model_trainer.train_and_evaluate(
        X_train, y_train,
        X_test, y_test,
        baseline_cost
    )
    return results


def run_optuna_optimization(
    model_type: str,
    class_weight_ratio: float,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_trials: int = 50,
    optimize_by: str = 'f1',
    baseline_cost: float = None,
    use_early_stopping: bool = True,
    early_stopping_rounds: int = 50
) -> Tuple[OptunaOptimizer, optuna.Study]:
    """Ejecuta la optimización con Optuna y muestra resultados.
    
    Args:
        model_type: Tipo de modelo ('XGBoost', 'LightGBM', 'CatBoost')
        class_weight_ratio: Ratio de peso de clases
        X_train, y_train: Datos de entrenamiento
        X_test, y_test: Datos de test
        n_trials: Número de trials de Optuna
        optimize_by: Métrica a optimizar ('cost', 'f1', 'recall', 'auc')
        baseline_cost: Costo baseline (requerido si optimize_by='cost')
        use_early_stopping: Si True, usa early stopping durante el entrenamiento
        early_stopping_rounds: Número de rondas sin mejora para detener
    """
    print_section(f"OPTIMIZACIÓN DE HIPERPARÁMETROS CON OPTUNA - {model_type}")
    
    optimizer = OptunaOptimizer(
        model_type=model_type,
        class_weight_ratio=class_weight_ratio,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        optimize_by=optimize_by,
        baseline_cost=baseline_cost,
        use_early_stopping=use_early_stopping,
        early_stopping_rounds=early_stopping_rounds
    )
    
    study = optimizer.optimize(n_trials=n_trials)
    
    print(f"\n✅ Optimización completada!")
    metric_names = {'cost': 'Costo', 'f1': 'F1-Score', 'recall': 'Recall', 'auc': 'AUC'}
    print(f"\n Mejor {metric_names.get(optimize_by, optimize_by)} encontrado: {study.best_value:.4f}")
    print(f"\n Mejores hiperparámetros:")
    for key, value in study.best_params.items():
        print(f"   - {key}: {value}")
        
    return optimizer, study


def run_threshold_optimization(
    y_test: np.ndarray,
    y_pred_proba: np.ndarray,
    baseline_cost: float,
    optimize_by: str = 'f1'
) -> Tuple[float, pd.DataFrame]:
    """Ejecuta optimización de umbral y muestra análisis."""
    optimal_threshold, threshold_df = optimize_threshold(
        y_test, y_pred_proba, baseline_cost, optimize_by=optimize_by
    )
    optimal_cost = threshold_df.loc[threshold_df['threshold'] == optimal_threshold, 'cost'].values[0]
    optimal_savings = baseline_cost - optimal_cost
    
    analyze_threshold_optimization(optimal_threshold, optimal_cost, optimal_savings, baseline_cost)
    
    return optimal_threshold, threshold_df


def evaluate_final_model_performance(
    y_test: np.ndarray,
    y_pred_proba: np.ndarray,
    optimal_threshold: float,
    baseline_cost: float,
    model_name: str
) -> Tuple[np.ndarray, float]:
    """Evalúa el modelo final con el umbral óptimo."""
    y_pred_final = (y_pred_proba >= optimal_threshold).astype(int)
    final_cost, _ = calculate_cost(y_test, y_pred_final)
    
    analyze_final_model(y_test, y_pred_final, model_name, optimal_threshold, baseline_cost)
    
    return y_pred_final, final_cost


def select_best_boost_model(results: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Selecciona el mejor modelo de boosting para optimizar."""
    print_section("SELECCIÓN DEL MODELO GANADOR PARA OPTIMIZACIÓN")
    
    boost_models = {k: v for k, v in results.items() if k in ['XGBoost', 'LightGBM', 'CatBoost']}
    if not boost_models:
        raise ValueError("No se encontraron modelos XGBoost, LightGBM o CatBoost en los resultados.")
        
    best_boost_name = max(boost_models, key=lambda x: results[x]['f1'])
    best_boost = results[best_boost_name]

    print(f"\nComparación XGBoost vs LightGBM vs CatBoost (por F1):")
    for name in boost_models:
        print(f"  - {name} F1: {results[name]['f1']:.4f}")
    print(f"\n Modelo ganador para optimizar: {best_boost_name}")
    
    return best_boost_name, best_boost


def analyze_optimized_model_performance(
    optimized_model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    baseline_cost: float,
    best_boost_before: Dict[str, Any],
    best_boost_name: str
) -> Dict[str, Any]:
    """Calcula y muestra métricas del modelo optimizado comparado con el anterior."""
    
    y_pred_optimized = optimized_model.predict(X_test)
    y_pred_proba_optimized = optimized_model.predict_proba(X_test)[:, 1]

    # Métricas del modelo optimizado
    auc_optimized = roc_auc_score(y_test, y_pred_proba_optimized)
    precision_optimized = precision_score(y_test, y_pred_optimized)
    recall_optimized = recall_score(y_test, y_pred_optimized)
    f1_optimized = f1_score(y_test, y_pred_optimized)
    total_cost_optimized, _ = calculate_cost(y_test, y_pred_optimized)
    savings_optimized = baseline_cost - total_cost_optimized

    print(f"\n✅ Modelo optimizado!")
    print(f"\n COMPARACIÓN: Antes vs Después de Optuna")
    print("-" * 50)
    print(f"{'Métrica':<25} {'Antes':<15} {'Después':<15}")
    print("-" * 50)
    print(f"{'F1-Score':<25} {best_boost_before['f1']:<15.4f} {f1_optimized:<15.4f}")
    print(f"{'Recall':<25} {best_boost_before['recall']:<15.4f} {recall_optimized:<15.4f}")
    print(f"{'Precision':<25} {best_boost_before['precision']:<15.4f} {precision_optimized:<15.4f}")
    print(f"{'AUC-ROC':<25} {best_boost_before['auc']:<15.4f} {auc_optimized:<15.4f}")
    print(f"{'Costo Total':<25} ${best_boost_before['total_cost']:<14,.2f} ${total_cost_optimized:<14,.2f}")

    # Estructura del mejor modelo
    return {
        'model': optimized_model,
        'y_pred': y_pred_optimized,
        'y_pred_proba': y_pred_proba_optimized,
        'auc': auc_optimized,
        'precision': precision_optimized,
        'recall': recall_optimized,
        'f1': f1_optimized,
        'total_cost': total_cost_optimized,
        'savings': savings_optimized,
        'savings_pct': (savings_optimized / baseline_cost) * 100,
        'name_optimized': f"{best_boost_name} (Optimizado)"
    }


def analyze_baseline_costs(y_test: np.ndarray) -> Tuple[float, float]:
    """Analiza y muestra costos baseline."""
    print_section("ANÁLISIS DE COSTOS BASELINE")
    
    # Matriz de costos
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│                        MATRIZ DE COSTOS                             │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print("│                    │ Predicho: No Falla │ Predicho: Falla           │")
    print("├────────────────────┼────────────────────┼───────────────────────────┤")
    print(f"│ Real: No Falla     │ TN = $0 (óptimo)   │ FP = ${COST_MAINTENANCE} (mant. innecesario) │")
    print(f"│ Real: Falla        │ FN = ${COST_FAILURE} (no prevenida)│ TP = ${COST_MAINTENANCE} (mant. previene)    │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    
    n_failures = int(y_test.sum())
    n_total = len(y_test)
    n_no_failures = n_total - n_failures
    
    # Escenarios
    baseline = calculate_baseline_cost(y_test)  # No hacer nada: todas las fallas ocurren
    optimal = n_failures * COST_MAINTENANCE  # Óptimo: mantener solo los que van a fallar
    
    print(f"\n📊 ESCENARIOS DE COSTO (Test: {n_total:,} muestras, {n_failures:,} fallas)")
    print("-" * 60)
    print(f"  1. No hacer nada (todas fallas ocurren):     ${baseline:,.2f}")
    print(f"     → {n_failures} FN × ${COST_FAILURE} = ${baseline:,.2f}")
    print(f"\n  2. ÓPTIMO (mantener solo los que fallan):    ${optimal:,.2f}")
    print(f"     → {n_failures} TP × ${COST_MAINTENANCE} = ${optimal:,.2f}")
    
    print("\n" + "-" * 60)
    print(f"Meta del modelo: acercarse al óptimo de ${optimal:,.2f}")
    
    return baseline


def analyze_split(
    X_train: pd.DataFrame, X_test: pd.DataFrame,
    y_train: pd.Series, y_test: pd.Series,
    split_date: pd.Timestamp
) -> None:
    """Muestra información del split de datos."""
    print_section("SPLIT DE DATOS")
    print(f"\nSplit temporal en: {split_date.strftime('%Y-%m-%d')}")
    print(f"\nTrain: {len(X_train):,} muestras")
    print(f"  - No fallas: {(y_train == 0).sum():,} ({(y_train == 0).mean()*100:.2f}%)")
    print(f"  - Fallas: {(y_train == 1).sum():,} ({(y_train == 1).mean()*100:.2f}%)")
    print(f"\nTest: {len(X_test):,} muestras")
    print(f"  - No fallas: {(y_test == 0).sum():,} ({(y_test == 0).mean()*100:.2f}%)")
    print(f"  - Fallas: {(y_test == 1).sum():,} ({(y_test == 1).mean()*100:.2f}%)")


def analyze_smote(y_before: np.ndarray, y_after: np.ndarray) -> None:
    """Muestra información del balanceo con SMOTE."""
    print_section("BALANCEO CON SMOTE")
    print(f"\nAntes: {len(y_before):,} muestras")
    print(f"  - No fallas: {(y_before == 0).sum():,}")
    print(f"  - Fallas: {(y_before == 1).sum():,}")
    print(f"\nDespués: {len(y_after):,} muestras")
    print(f"  - No fallas: {(y_after == 0).sum():,}")
    print(f"  - Fallas: {(y_after == 1).sum():,}")


def analyze_threshold_optimization(
    optimal_threshold: float,
    optimal_cost: float,
    optimal_savings: float,
    baseline_cost: float
) -> None:
    """Muestra resultados de optimización de umbral."""
    print_section("OPTIMIZACIÓN DEL UMBRAL")
    print(f"\nUmbral óptimo: {optimal_threshold:.2f}")
    print(f"Costo con umbral óptimo: ${optimal_cost:,.2f}")
    print(f"Ahorro: ${optimal_savings:,.2f} ({optimal_savings/baseline_cost*100:.2f}%)")


def analyze_final_model(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    optimal_threshold: float,
    baseline_cost: float
) -> None:
    """Muestra análisis completo del modelo final."""
    from sklearn.metrics import classification_report
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    final_cost, _ = calculate_cost(y_test, y_pred)
    
    print_section(f"MODELO FINAL: {model_name}")
    print(f"Umbral: {optimal_threshold:.2f}")
    
    print(f"\n📊 MÉTRICAS:")
    print(classification_report(y_test, y_pred, target_names=['No Falla', 'Falla']))
    
    print(f"📈 MATRIZ DE CONFUSIÓN:")
    print(f"                Predicho")
    print(f"                No Falla    Falla")
    print(f"Real No Falla   {cm[0,0]:>8,}  {cm[0,1]:>8,}")
    print(f"Real Falla      {cm[1,0]:>8,}  {cm[1,1]:>8,}")
    
    print(f"\n💰 ANÁLISIS DE COSTOS:")
    print(f"  - TN (óptimo):           {tn:,} → $0")
    print(f"  - FP (mant. innecesario): {fp:,} → ${fp * COST_MAINTENANCE:,.2f}")
    print(f"  - FN (falla no prevenida): {fn:,} → ${fn * COST_FAILURE:,.2f}")
    print(f"  - TP (mant. previene):    {tp:,} → ${tp * COST_MAINTENANCE:,.2f}")
    print(f"\n  - Costo total:  ${final_cost:,.2f}")
    print(f"  - Baseline:     ${baseline_cost:,.2f}")
    print(f"  - AHORRO:       ${baseline_cost - final_cost:,.2f} ({(baseline_cost - final_cost)/baseline_cost*100:.2f}%)")


def print_executive_summary(
    df: pd.DataFrame,
    feature_cols: List[str],
    y: pd.Series,
    model_name: str,
    optimal_threshold: float,
    best_f1: float,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    baseline_cost: float,
    final_cost: float
) -> None:
    """Imprime resumen ejecutivo del proyecto."""
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print_header("RESUMEN EJECUTIVO", "=", 70)
    print(f"""
PROBLEMA
   Predecir fallas en dispositivos para optimizar costos de mantenimiento.

DATOS
   - {df.shape[0]:,} registros de telemetría
   - {df['device'].nunique():,} dispositivos únicos
   - {len(feature_cols)} features engineered
   - Período: {df['date'].min().strftime('%Y-%m-%d')} a {df['date'].max().strftime('%Y-%m-%d')}
   - Tasa de fallas: {(y.sum()/len(y)*100):.2f}%

MEJOR MODELO: {model_name}
   - Umbral óptimo: {optimal_threshold:.2f}
   - F1-Score: {best_f1:.4f}
   - Recall: {recall_score(y_test, y_pred):.4f}
   - Precision: {precision_score(y_test, y_pred):.4f}

IMPACTO FINANCIERO
   - Costo sin modelo: ${baseline_cost:,.2f}
   - Costo con modelo: ${final_cost:,.2f}
   - AHORRO TOTAL:     ${baseline_cost - final_cost:,.2f} ({(baseline_cost - final_cost)/baseline_cost*100:.2f}%)

EFICIENCIA
   - Fallas detectadas: {tp:,}/{y_test.sum():,} ({tp/y_test.sum()*100:.1f}%)
   - Fallas no detectadas: {fn:,}
   - Mantenimientos innecesarios: {fp:,}

RECOMENDACIONES
   1. Implementar modelo con umbral {optimal_threshold:.2f}
   2. Monitorear métricas semanalmente
   3. Reentrenar mensualmente
""")
    print("=" * 70)


# =============================================================================
# FUNCIONES ESPECÍFICAS PARA V5 - NOTEBOOK DE PRESENTACIÓN
# =============================================================================

# Constantes V5
V5_KEY_ATTRS = ['attribute7', 'attribute4', 'attribute2']

# Configuración óptima encontrada por Optuna en V5
V5_OPTIMAL_CONFIG = {
    'model_type': 'easyensemble',
    'n_estimators': 24,
    'base_n_estimators': 52,
    'base_learning_rate': 0.056,
    'threshold': 0.87,
    # Features seleccionadas
    'use_rolling_mean': True,
    'use_rolling_std': True,
    'use_diff': True,
    'use_spike': False,
    'use_max': True,
    'use_zscore': True,
    'use_pct': True,
    'use_interaction': True,
    'use_min_ratio': True,
    'use_days_since': True,
    'use_momentum': True,
    'use_bollinger': True,
    'use_interaction_v5': True
}


def load_data_v5(data_path: str = None) -> pd.DataFrame:
    """
    Carga el dataset de dispositivos.
    
    Args:
        data_path: Ruta al archivo CSV (opcional)
        
    Returns:
        DataFrame con los datos
    """
    import os
    if data_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, 'full_devices.csv')
    
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print_section("DATOS CARGADOS")
    print(f"  Registros: {len(df):,}")
    print(f"  Dispositivos: {df['device'].nunique():,}")
    print(f"  Fallas: {df['failure'].sum()} ({df['failure'].mean()*100:.3f}%)")
    print(f"  Período: {df['date'].min().strftime('%Y-%m-%d')} a {df['date'].max().strftime('%Y-%m-%d')}")
    
    return df


def create_features_v5(df: pd.DataFrame, attrs: List[str] = None) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Crea TODAS las features V5 (incluyendo las avanzadas).
    
    Args:
        df: DataFrame con los datos
        attrs: Lista de atributos a usar (default: V5_KEY_ATTRS)
        
    Returns:
        Tuple con (DataFrame con features, diccionario de grupos de features)
    """
    if attrs is None:
        attrs = V5_KEY_ATTRS
    
    df = df.copy().sort_values(['device', 'date'])
    feature_groups = {}
    
    print_section("CREANDO FEATURES V5")
    print(f"  Atributos base: {attrs}")
    
    # =========================================================================
    # FEATURES V4 (base)
    # =========================================================================
    
    # Rolling means
    rolling_mean_cols = []
    for attr in attrs:
        for w in [2, 3, 5, 7, 10, 14]:
            col = f'{attr}_roll_mean_{w}d'
            df[col] = df.groupby('device')[attr].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            )
            rolling_mean_cols.append(col)
    feature_groups['rolling_mean'] = rolling_mean_cols
    
    # Rolling std
    rolling_std_cols = []
    for attr in attrs:
        for w in [3, 7, 14]:
            col = f'{attr}_roll_std_{w}d'
            df[col] = df.groupby('device')[attr].transform(
                lambda x: x.shift(1).rolling(w, min_periods=2).std()
            )
            rolling_std_cols.append(col)
    feature_groups['rolling_std'] = rolling_std_cols
    
    # Diff y aceleración
    diff_cols = []
    for attr in attrs:
        df[f'{attr}_diff'] = df.groupby('device')[attr].diff()
        diff_cols.append(f'{attr}_diff')
        df[f'{attr}_accel'] = df.groupby('device')[f'{attr}_diff'].diff()
        diff_cols.append(f'{attr}_accel')
    feature_groups['diff'] = diff_cols
    
    # Spike ratios
    spike_cols = []
    for attr in attrs:
        for w in [3, 5, 7, 14]:
            rolling_mean = df.groupby('device')[attr].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            )
            col = f'{attr}_spike_{w}d'
            df[col] = df[attr] / (rolling_mean + 1e-6)
            spike_cols.append(col)
    feature_groups['spike'] = spike_cols
    
    # Ratio al máximo
    max_cols = []
    for attr in attrs:
        df[f'{attr}_ratio_max'] = df.groupby('device')[attr].transform(
            lambda x: x / (x.shift(1).expanding().max() + 1e-6)
        )
        max_cols.append(f'{attr}_ratio_max')
    feature_groups['max'] = max_cols
    
    # Z-score
    zscore_cols = []
    for attr in attrs:
        device_mean = df.groupby('device')[attr].transform(lambda x: x.shift(1).expanding().mean())
        device_std = df.groupby('device')[attr].transform(lambda x: x.shift(1).expanding().std())
        col = f'{attr}_zscore'
        df[col] = (df[attr] - device_mean) / (device_std + 1e-6)
        zscore_cols.append(col)
    feature_groups['zscore'] = zscore_cols
    
    # Percentil
    pct_cols = []
    for attr in attrs:
        col = f'{attr}_pct_rank'
        df[col] = df.groupby('device')[attr].transform(lambda x: x.rank(pct=True))
        pct_cols.append(col)
    feature_groups['pct'] = pct_cols
    
    # Interacciones V4
    interaction_cols = []
    df['attr7_x_attr4'] = df['attribute7'] * df['attribute4']
    interaction_cols.append('attr7_x_attr4')
    if 'attribute7_spike_7d' in df.columns:
        df['spike7_x_spike4'] = df['attribute7_spike_7d'] * df['attribute4_spike_7d']
        interaction_cols.append('spike7_x_spike4')
    feature_groups['interaction'] = interaction_cols
    
    # Device age
    df['device_age'] = (df['date'] - df.groupby('device')['date'].transform('min')).dt.days
    feature_groups['device_age'] = ['device_age']
    
    # =========================================================================
    # FEATURES V5 (avanzadas)
    # =========================================================================
    
    # Ratio al mínimo histórico
    min_cols = []
    for attr in attrs:
        df[f'{attr}_ratio_min'] = df.groupby('device')[attr].transform(
            lambda x: x / (x.shift(1).expanding().min() + 1e-6)
        )
        min_cols.append(f'{attr}_ratio_min')
        hist_min = df.groupby('device')[attr].transform(lambda x: x.shift(1).expanding().min())
        hist_max = df.groupby('device')[attr].transform(lambda x: x.shift(1).expanding().max())
        df[f'{attr}_range_pos'] = (df[attr] - hist_min) / (hist_max - hist_min + 1e-6)
        min_cols.append(f'{attr}_range_pos')
    feature_groups['min_ratio'] = min_cols
    
    # Días desde último máximo
    days_since_cols = []
    for attr in attrs:
        def days_since_max(x):
            result = pd.Series(index=x.index, dtype=float)
            running_max = -np.inf
            days_since = 0
            for i, val in enumerate(x):
                if i == 0:
                    result.iloc[i] = 0
                    running_max = val
                else:
                    if x.iloc[i-1] >= running_max:
                        running_max = x.iloc[i-1]
                        days_since = 1
                    else:
                        days_since += 1
                    result.iloc[i] = days_since
            return result
        col = f'{attr}_days_since_max'
        df[col] = df.groupby('device')[attr].transform(days_since_max)
        days_since_cols.append(col)
    feature_groups['days_since'] = days_since_cols
    
    # Momentum / Rate of Change
    momentum_cols = []
    for attr in attrs:
        for w in [3, 7, 14]:
            past_val = df.groupby('device')[attr].shift(w)
            col = f'{attr}_roc_{w}d'
            df[col] = (df[attr] - past_val) / (past_val + 1e-6)
            momentum_cols.append(col)
    feature_groups['momentum'] = momentum_cols
    
    # Bollinger bands
    bollinger_cols = []
    for attr in attrs:
        roll_mean = df.groupby('device')[attr].transform(
            lambda x: x.shift(1).rolling(14, min_periods=2).mean()
        )
        roll_std = df.groupby('device')[attr].transform(
            lambda x: x.shift(1).rolling(14, min_periods=2).std()
        )
        upper_band = roll_mean + 2 * roll_std
        lower_band = roll_mean - 2 * roll_std
        col = f'{attr}_bollinger_pos'
        df[col] = (df[attr] - lower_band) / (upper_band - lower_band + 1e-6)
        bollinger_cols.append(col)
        df[f'{attr}_above_upper'] = (df[attr] > upper_band).astype(int)
        bollinger_cols.append(f'{attr}_above_upper')
    feature_groups['bollinger'] = bollinger_cols
    
    # Interacciones V5
    interaction_v5_cols = []
    if 'attribute7_zscore' in df.columns:
        df['zscore7_x_zscore4'] = df['attribute7_zscore'] * df['attribute4_zscore']
        interaction_v5_cols.append('zscore7_x_zscore4')
        df['zscore_sum'] = df['attribute7_zscore'] + df['attribute4_zscore'] + df['attribute2_zscore']
        interaction_v5_cols.append('zscore_sum')
        df['zscore_max'] = df[['attribute7_zscore', 'attribute4_zscore', 'attribute2_zscore']].max(axis=1)
        interaction_v5_cols.append('zscore_max')
    above_upper_cols = [c for c in df.columns if 'above_upper' in c]
    if len(above_upper_cols) == 3:
        df['n_above_upper'] = df[above_upper_cols].sum(axis=1)
        interaction_v5_cols.append('n_above_upper')
    feature_groups['interaction_v5'] = interaction_v5_cols
    
    # Limpiar
    all_cols = [c for cols in feature_groups.values() for c in cols]
    df[all_cols] = df[all_cols].fillna(0).replace([np.inf, -np.inf], 0)
    
    # Resumen
    total_features = sum(len(cols) for cols in feature_groups.values())
    print(f"\n  Features generadas por grupo:")
    for name, cols in feature_groups.items():
        print(f"    - {name}: {len(cols)}")
    print(f"  Total features: {total_features}")
    
    return df, feature_groups


def select_features_v5(
    feature_groups: Dict[str, List[str]], 
    config: Dict[str, Any] = None
) -> List[str]:
    """
    Selecciona features según la configuración óptima de V5.
    
    Args:
        feature_groups: Diccionario con grupos de features
        config: Configuración de features a usar (default: V5_OPTIMAL_CONFIG)
        
    Returns:
        Lista de features seleccionadas
    """
    if config is None:
        config = V5_OPTIMAL_CONFIG
    
    selected = V5_KEY_ATTRS.copy()
    selected.append('device_age')
    
    # Mapeo de configuración a grupos
    group_mapping = {
        'use_rolling_mean': 'rolling_mean',
        'use_rolling_std': 'rolling_std',
        'use_diff': 'diff',
        'use_spike': 'spike',
        'use_max': 'max',
        'use_zscore': 'zscore',
        'use_pct': 'pct',
        'use_interaction': 'interaction',
        'use_min_ratio': 'min_ratio',
        'use_days_since': 'days_since',
        'use_momentum': 'momentum',
        'use_bollinger': 'bollinger',
        'use_interaction_v5': 'interaction_v5'
    }
    
    for use_key, group_name in group_mapping.items():
        if config.get(use_key, False) and group_name in feature_groups:
            selected.extend(feature_groups[group_name])
    
    return selected


def prepare_data_v5(
    df: pd.DataFrame, 
    features: List[str], 
    test_size: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Prepara los datos con split temporal 80/20.
    
    Args:
        df: DataFrame con features
        features: Lista de features a usar
        test_size: Proporción de test (default: 0.2)
        
    Returns:
        Tuple con (X_train_scaled, X_test_scaled, y_train, y_test, df_train, df_test)
    """
    df = df.sort_values('date')
    split = int(len(df) * (1 - test_size))
    
    df_train = df.iloc[:split].copy()
    df_test = df.iloc[split:].copy()
    
    X_train, X_test = df[features].iloc[:split], df[features].iloc[split:]
    y_train, y_test = df['failure'].iloc[:split].values, df['failure'].iloc[split:].values
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print_section("SPLIT TEMPORAL DE DATOS")
    print(f"  Train: {len(y_train):,} registros ({y_train.sum()} fallas)")
    print(f"  Test:  {len(y_test):,} registros ({y_test.sum()} fallas)")
    print(f"  Features: {len(features)}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, df_train, df_test


def train_model_v5(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    config: Dict[str, Any] = None
) -> Any:
    """
    Entrena el modelo V5 con la configuración óptima.
    
    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        config: Configuración del modelo (default: V5_OPTIMAL_CONFIG)
        
    Returns:
        Modelo entrenado
    """
    from imblearn.ensemble import EasyEnsembleClassifier
    from sklearn.ensemble import AdaBoostClassifier
    
    if config is None:
        config = V5_OPTIMAL_CONFIG
    
    print_section("ENTRENANDO MODELO V5")
    print(f"  Modelo: EasyEnsembleClassifier")
    print(f"  n_estimators: {config['n_estimators']}")
    print(f"  base_n_estimators: {config['base_n_estimators']}")
    print(f"  base_learning_rate: {config['base_learning_rate']}")
    
    model = EasyEnsembleClassifier(
        n_estimators=config['n_estimators'],
        estimator=AdaBoostClassifier(
            n_estimators=config['base_n_estimators'],
            learning_rate=config['base_learning_rate'],
            random_state=42
        ),
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print(f"  ✅ Modelo entrenado!")
    
    return model


def predict_v5(
    model: Any, 
    X_test: np.ndarray, 
    threshold: float = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Realiza predicciones con el modelo V5. (Wrapper de predict_with_threshold)
    
    Returns:
        Tuple con (predicciones binarias, probabilidades) - Nota: orden inverso a predict_final
    """
    if threshold is None:
        threshold = V5_OPTIMAL_CONFIG['threshold']
    y_proba, y_pred = predict_with_threshold(model, X_test, threshold)
    return y_pred, y_proba  # V5 retorna en orden inverso por compatibilidad


def calculate_cost_v5(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, Dict[str, int]]:
    """
    Calcula el costo y métricas del modelo.
    
    Args:
        y_true: Valores reales
        y_pred: Predicciones
        
    Returns:
        Tuple con (costo total, diccionario con TP, FP, FN, TN)
    """
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = cm[0][0], 0, int(y_true.sum()), 0
    
    cost = tp * COST_MAINTENANCE + fp * COST_MAINTENANCE + fn * COST_FAILURE
    
    return cost, {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn}


def print_metrics_v5(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    threshold: float = None,
    model_name: str = "V5"
) -> Dict[str, Any]:
    """
    Imprime las métricas del modelo V5 de forma visual.
    
    Args:
        y_true: Valores reales
        y_pred: Predicciones
        threshold: Umbral usado
        model_name: Nombre del modelo
        
    Returns:
        Diccionario con todas las métricas
    """
    if threshold is None:
        threshold = V5_OPTIMAL_CONFIG['threshold']
    
    cost, metrics = calculate_cost_v5(y_true, y_pred)
    baseline = y_true.sum() * COST_FAILURE
    ahorro = (baseline - cost) / baseline * 100
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    MÉTRICAS MODELO {model_name:^10}                        ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Threshold: {threshold:.2f}                                                   ║
║                                                                      ║
║  Detección de Fallas:                                                ║
║  • True Positives (fallas detectadas):    {metrics['TP']:>3} de {int(y_true.sum())}             ║
║  • False Negatives (fallas perdidas):     {metrics['FN']:>3}                            ║
║  • Recall: {recall:.1%}                                                    ║
║                                                                      ║
║  Precisión de Alertas:                                               ║
║  • False Positives (falsas alarmas):      {metrics['FP']:>3}                            ║
║  • Precision: {precision:.1%}                                                 ║
║                                                                      ║
║  Análisis de Costos:                                                 ║
║  • Baseline (no hacer nada):              ${baseline:>6.1f}                     ║
║  • Costo con modelo:                      ${cost:>6.1f}                     ║
║  • Ahorro:                                ${baseline - cost:>6.1f} ({ahorro:>5.1f}%)           ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    return {
        'cost': cost,
        'baseline': baseline,
        'ahorro_pct': ahorro,
        'precision': precision,
        'recall': recall,
        **metrics
    }


def plot_confusion_matrix_v5(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    title: str = "Matriz de Confusión con Costos"
) -> None:
    """
    Grafica la matriz de confusión con costos anotados.
    
    Args:
        y_true: Valores reales
        y_pred: Predicciones
        title: Título del gráfico
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Crear matriz con anotaciones de costo
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Datos para el heatmap
    data = np.array([[tn, fp], [fn, tp]])
    
    # Crear heatmap
    im = ax.imshow(data, cmap='Blues')
    
    # Etiquetas
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred: No Falla', 'Pred: Falla'], fontsize=12)
    ax.set_yticklabels(['Real: No Falla', 'Real: Falla'], fontsize=12)
    
    # Anotaciones detalladas
    annotations = [
        [f'TN: {tn:,}\n$0', f'FP: {fp:,}\n${fp * COST_MAINTENANCE:.1f}'],
        [f'FN: {fn:,}\n${fn * COST_FAILURE:.1f}', f'TP: {tp:,}\n${tp * COST_MAINTENANCE:.1f}']
    ]
    
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, annotations[i][j],
                          ha="center", va="center", fontsize=14, fontweight='bold',
                          color="white" if data[i, j] > data.max()/2 else "black")
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    plt.colorbar(im, ax=ax, label='Cantidad')
    plt.tight_layout()
    plt.show()


def plot_precision_recall_v5(
    y_true: np.ndarray, 
    y_proba: np.ndarray,
    optimal_threshold: float = None
) -> None:
    """
    Grafica la curva Precision-Recall con el threshold óptimo marcado.
    
    Args:
        y_true: Valores reales
        y_proba: Probabilidades predichas
        optimal_threshold: Umbral óptimo a marcar
    """
    if optimal_threshold is None:
        optimal_threshold = V5_OPTIMAL_CONFIG['threshold']
    
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Curva PR
    ax.plot(recall_curve, precision_curve, 'b-', linewidth=2, 
            label=f'Modelo V5 (AP={avg_precision:.3f})')
    
    # Línea base (precision = % de fallas)
    baseline = y_true.mean()
    ax.axhline(y=baseline, color='gray', linestyle='--', 
               label=f'Baseline ({baseline:.3%})')
    
    # Marcar threshold óptimo
    idx = np.argmin(np.abs(thresholds - optimal_threshold))
    ax.scatter(recall_curve[idx], precision_curve[idx], 
               s=200, c='red', marker='*', zorder=5,
               label=f'Threshold={optimal_threshold:.2f}')
    
    # Línea del 50% de precision
    ax.axhline(y=0.5, color='orange', linestyle=':', alpha=0.7,
               label='50% Precision (break-even)')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Curva Precision-Recall', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.show()


def plot_feature_importance_v5(
    model: Any,
    feature_names: List[str],
    top_k: int = 20
) -> None:
    """
    Grafica la importancia de features del modelo V5.
    
    Args:
        model: Modelo entrenado (EasyEnsembleClassifier)
        feature_names: Nombres de las features
        top_k: Número de features a mostrar
    """
    # EasyEnsemble tiene múltiples estimadores base
    # Promediamos las importancias
    importances = np.zeros(len(feature_names))
    
    for estimator in model.estimators_:
        # Cada estimator es un AdaBoost
        for ada_estimator, weight in zip(estimator.estimators_, estimator.estimator_weights_):
            if hasattr(ada_estimator, 'feature_importances_'):
                importances += ada_estimator.feature_importances_ * weight
    
    importances /= len(model.estimators_)
    
    # Crear DataFrame y ordenar
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=True).tail(top_k)
    
    # Graficar
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(importance_df)))
    bars = ax.barh(importance_df['feature'], importance_df['importance'], color=colors)
    
    ax.set_xlabel('Importancia', fontsize=12)
    ax.set_title(f'Top {top_k} Features Más Importantes', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()


def plot_threshold_analysis_v5(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    optimal_threshold: float = None
) -> pd.DataFrame:
    """
    Analiza el impacto del threshold en costo, precision y recall.
    
    Args:
        y_true: Valores reales
        y_proba: Probabilidades predichas
        optimal_threshold: Umbral óptimo a marcar
        
    Returns:
        DataFrame con análisis por threshold
    """
    if optimal_threshold is None:
        optimal_threshold = V5_OPTIMAL_CONFIG['threshold']
    
    thresholds = np.arange(0.1, 1.0, 0.05)
    results = []
    baseline = y_true.sum() * COST_FAILURE
    
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        cost, metrics = calculate_cost_v5(y_true, y_pred)
        
        results.append({
            'threshold': t,
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'cost': cost,
            'ahorro_pct': (baseline - cost) / baseline * 100,
            'TP': metrics['TP'],
            'FP': metrics['FP']
        })
    
    df_results = pd.DataFrame(results)
    
    # Graficar
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 1. Precision vs Recall
    ax1 = axes[0]
    ax1.plot(df_results['threshold'], df_results['precision'], 'b-', linewidth=2, label='Precision')
    ax1.plot(df_results['threshold'], df_results['recall'], 'g-', linewidth=2, label='Recall')
    ax1.axvline(x=optimal_threshold, color='red', linestyle='--', label=f'Óptimo={optimal_threshold:.2f}')
    ax1.axhline(y=0.5, color='orange', linestyle=':', alpha=0.7, label='50% (break-even)')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Valor')
    ax1.set_title('Precision vs Recall')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Costo vs Threshold
    ax2 = axes[1]
    ax2.plot(df_results['threshold'], df_results['cost'], 'r-', linewidth=2)
    ax2.axvline(x=optimal_threshold, color='red', linestyle='--')
    ax2.axhline(y=baseline, color='gray', linestyle=':', label=f'Baseline=${baseline:.0f}')
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Costo ($)')
    ax2.set_title('Costo vs Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Ahorro vs Threshold
    ax3 = axes[2]
    ax3.fill_between(df_results['threshold'], df_results['ahorro_pct'], alpha=0.3, color='green')
    ax3.plot(df_results['threshold'], df_results['ahorro_pct'], 'g-', linewidth=2)
    ax3.axvline(x=optimal_threshold, color='red', linestyle='--', label=f'Óptimo={optimal_threshold:.2f}')
    ax3.axhline(y=0, color='gray', linestyle='-')
    ax3.set_xlabel('Threshold')
    ax3.set_ylabel('Ahorro (%)')
    ax3.set_title('Ahorro vs Threshold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle('Análisis de Sensibilidad del Threshold', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return df_results


def plot_version_comparison() -> None:
    """
    Grafica la evolución de resultados a través de las versiones V1-V5.
    """
    versions = ['V1-V2', 'V3', 'V4', 'V5']
    recall = [4.3, 30.4, 39.1, 43.5]
    ahorro = [2.2, 4.3, 6.5, 8.7]
    fallas_detectadas = [1, 7, 9, 10]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Colores
    colors = ['#e74c3c', '#f39c12', '#3498db', '#27ae60']
    
    # 1. Recall por versión
    ax1 = axes[0]
    bars1 = ax1.bar(versions, recall, color=colors)
    ax1.set_ylabel('Recall (%)')
    ax1.set_title('Evolución del Recall')
    ax1.set_ylim(0, 50)
    for bar, val in zip(bars1, recall):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val}%', ha='center', fontweight='bold')
    
    # 2. Ahorro por versión
    ax2 = axes[1]
    bars2 = ax2.bar(versions, ahorro, color=colors)
    ax2.set_ylabel('Ahorro (%)')
    ax2.set_title('Evolución del Ahorro')
    ax2.set_ylim(0, 12)
    for bar, val in zip(bars2, ahorro):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                f'+{val}%', ha='center', fontweight='bold')
    
    # 3. Fallas detectadas
    ax3 = axes[2]
    bars3 = ax3.bar(versions, fallas_detectadas, color=colors)
    ax3.axhline(y=23, color='gray', linestyle='--', label='Total fallas (23)')
    ax3.set_ylabel('Fallas Detectadas')
    ax3.set_title('Fallas Detectadas de 23')
    ax3.set_ylim(0, 25)
    ax3.legend()
    for bar, val in zip(bars3, fallas_detectadas):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val}', ha='center', fontweight='bold')
    
    plt.suptitle('Evolución del Modelo: V1 → V5', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Tabla resumen
    print("""
╔═══════════╦═════════════╦══════════════╦═══════════════╦═══════════════════════════════════╗
║  Versión  ║   Recall    ║    Ahorro    ║  TP (de 23)   ║           Técnica Clave           ║
╠═══════════╬═════════════╬══════════════╬═══════════════╬═══════════════════════════════════╣
║   V1-V2   ║    4.3%     ║    +2.2%     ║       1       ║  Class weights + SMOTE            ║
║    V3     ║   30.4%     ║    +4.3%     ║       7       ║  EasyEnsembleClassifier (7x!)     ║
║    V4     ║   39.1%     ║    +6.5%     ║       9       ║  Optuna (optimización bayesiana)  ║
║    V5     ║   43.5%     ║    +8.7%     ║      10       ║  Features avanzadas (bollinger)   ║
╚═══════════╩═════════════╩══════════════╩═══════════════╩═══════════════════════════════════╝
""")


def run_full_v5_pipeline(data_path: str = None) -> Dict[str, Any]:
    """
    Ejecuta el pipeline completo V5 y retorna todos los resultados.
    
    Args:
        data_path: Ruta al archivo de datos (opcional)
        
    Returns:
        Diccionario con modelo, predicciones y métricas
    """
    # 1. Cargar datos
    df = load_data_v5(data_path)
    
    # 2. Crear features
    df_features, feature_groups = create_features_v5(df)
    
    # 3. Seleccionar features
    selected_features = select_features_v5(feature_groups)
    
    # 4. Preparar datos
    X_train, X_test, y_train, y_test, df_train, df_test = prepare_data_v5(
        df_features, selected_features
    )
    
    # 5. Entrenar modelo
    model = train_model_v5(X_train, y_train)
    
    # 6. Predecir
    y_pred, y_proba = predict_v5(model, X_test)
    
    # 7. Calcular métricas
    metrics = print_metrics_v5(y_test, y_pred)
    
    return {
        'model': model,
        'df': df,
        'df_features': df_features,
        'df_train': df_train,
        'df_test': df_test,
        'feature_groups': feature_groups,
        'selected_features': selected_features,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'metrics': metrics
    }


# =============================================================================
# FUNCIONES PARA EDA - VENTANA DE PREDICCIÓN 7 DÍAS
# =============================================================================

def create_target_7d_window(df: pd.DataFrame, window_days: int = 7) -> pd.DataFrame:
    """
    Crea un target binario basado en ventana de predicción.
    
    Para cada registro de un dispositivo, el target es 1 si hay AL MENOS una falla
    en los próximos `window_days` días (incluyendo el día actual).
    
    Args:
        df: DataFrame con columnas 'date', 'device', 'failure'
        window_days: Número de días de la ventana de predicción (default: 7)
        
    Returns:
        DataFrame con columna adicional 'failure_within_Xd'
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['device', 'date'])
    
    # Para cada dispositivo, calcular si hay falla en los próximos N días
    def calc_failure_window(group):
        # Usamos rolling hacia adelante (shift negativo)
        # Rolling sum de las próximas window_days filas (incluyendo la actual)
        failure_ahead = group['failure'].iloc[::-1].rolling(window=window_days, min_periods=1).sum().iloc[::-1]
        return (failure_ahead > 0).astype(int)
    
    df[f'failure_within_{window_days}d'] = df.groupby('device', group_keys=False).apply(calc_failure_window)
    
    return df


def analyze_failure_window_distribution(df: pd.DataFrame, window_days: int = 7) -> Dict[str, Any]:
    """
    Analiza la distribución de clases con el target de ventana de predicción.
    
    Args:
        df: DataFrame con el target de ventana ya calculado
        window_days: Días de la ventana (para el nombre de columna)
        
    Returns:
        Diccionario con estadísticas del análisis
    """
    target_col = f'failure_within_{window_days}d'
    
    if target_col not in df.columns:
        df = create_target_7d_window(df, window_days)
    
    # Estadísticas generales
    total_records = len(df)
    positive_class = df[target_col].sum()
    negative_class = total_records - positive_class
    
    # Por dispositivo
    device_stats = df.groupby('device').agg({
        target_col: 'sum',
        'failure': 'sum',
        'date': 'count'
    }).rename(columns={'date': 'total_days', target_col: 'days_before_failure'})
    
    devices_with_warning = (device_stats['days_before_failure'] > 0).sum()
    devices_with_failure = (device_stats['failure'] > 0).sum()
    
    stats = {
        'total_records': total_records,
        'positive_class': positive_class,
        'negative_class': negative_class,
        'positive_rate': positive_class / total_records * 100,
        'imbalance_ratio': f"1:{int(negative_class / positive_class) if positive_class > 0 else 'inf'}",
        'total_devices': len(device_stats),
        'devices_with_failure': devices_with_failure,
        'devices_with_warning_days': devices_with_warning,
        'avg_warning_days_per_device': device_stats['days_before_failure'].mean(),
        'device_stats': device_stats
    }
    
    return stats


def get_devices_failure_timeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Obtiene el timeline de fallas por dispositivo: fechas de falla y días de observación.
    
    Args:
        df: DataFrame original con 'date', 'device', 'failure'
        
    Returns:
        DataFrame con información resumida por dispositivo
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    device_info = df.groupby('device').agg(
        first_observation=('date', 'min'),
        last_observation=('date', 'max'),
        total_days=('date', 'count'),
        total_failures=('failure', 'sum'),
        failure_dates=('date', lambda x: list(df.loc[x.index][df.loc[x.index, 'failure'] == 1]['date'].dt.strftime('%Y-%m-%d')))
    ).reset_index()
    
    device_info['days_span'] = (device_info['last_observation'] - device_info['first_observation']).dt.days + 1
    device_info['has_failure'] = device_info['total_failures'] > 0
    
    return device_info


def analyze_pre_failure_patterns(df: pd.DataFrame, days_before: int = 7) -> pd.DataFrame:
    """
    Analiza los patrones de atributos en los días previos a una falla.
    
    Para cada falla, extrae los registros de los `days_before` días anteriores
    para analizar señales precursoras.
    
    Args:
        df: DataFrame original
        days_before: Días antes de la falla a analizar
        
    Returns:
        DataFrame con los registros pre-falla y un indicador de días_hasta_falla
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['device', 'date'])
    
    pre_failure_records = []
    
    for device in df['device'].unique():
        device_df = df[df['device'] == device].copy()
        failure_dates = device_df[device_df['failure'] == 1]['date'].tolist()
        
        for failure_date in failure_dates:
            # Obtener registros de los días_before días anteriores
            start_date = failure_date - pd.Timedelta(days=days_before)
            window_df = device_df[
                (device_df['date'] >= start_date) & 
                (device_df['date'] <= failure_date)
            ].copy()
            
            window_df['days_to_failure'] = (failure_date - window_df['date']).dt.days
            window_df['failure_date'] = failure_date
            pre_failure_records.append(window_df)
    
    if pre_failure_records:
        return pd.concat(pre_failure_records, ignore_index=True)
    else:
        return pd.DataFrame()


def compare_attributes_before_failure(
    df: pd.DataFrame, 
    attributes: List[str] = None,
    days_before: int = 7
) -> pd.DataFrame:
    """
    Compara estadísticas de atributos entre días normales y días pre-falla.
    
    Args:
        df: DataFrame original
        attributes: Lista de atributos a comparar (default: attribute1-9)
        days_before: Días antes de falla para clasificar como "pre-falla"
        
    Returns:
        DataFrame con comparación estadística
    """
    if attributes is None:
        attributes = ATTRIBUTES
    
    # Crear target de ventana
    df_window = create_target_7d_window(df, days_before)
    target_col = f'failure_within_{days_before}d'
    
    # Calcular estadísticas para días normales vs pre-falla
    normal_days = df_window[df_window[target_col] == 0]
    pre_failure_days = df_window[df_window[target_col] == 1]
    
    comparison = []
    for attr in attributes:
        comparison.append({
            'attribute': attr,
            'normal_mean': normal_days[attr].mean(),
            'normal_std': normal_days[attr].std(),
            'pre_failure_mean': pre_failure_days[attr].mean(),
            'pre_failure_std': pre_failure_days[attr].std(),
            'mean_diff_pct': (pre_failure_days[attr].mean() - normal_days[attr].mean()) / 
                            (normal_days[attr].mean() + 1e-10) * 100
        })
    
    return pd.DataFrame(comparison)


def plot_failure_window_analysis(df: pd.DataFrame, window_days: int = 7) -> None:
    """
    Genera visualizaciones para el análisis de ventana de predicción de fallas.
    
    Args:
        df: DataFrame con datos originales
        window_days: Días de la ventana de predicción
    """
    # Crear target si no existe
    target_col = f'failure_within_{window_days}d'
    if target_col not in df.columns:
        df = create_target_7d_window(df, window_days)
    
    stats = analyze_failure_window_distribution(df, window_days)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Distribución de clases con ventana de 7 días
    ax1 = axes[0, 0]
    labels = [f'Sin falla próxima\n({stats["negative_class"]:,})', 
              f'Falla en {window_days}d\n({stats["positive_class"]:,})']
    colors = ['#2ecc71', '#e74c3c']
    ax1.bar(labels, [stats['negative_class'], stats['positive_class']], color=colors, edgecolor='black')
    ax1.set_title(f'Distribución de Clases - Ventana {window_days} días\n(Ratio: {stats["imbalance_ratio"]})')
    ax1.set_ylabel('Cantidad de registros')
    for i, v in enumerate([stats['negative_class'], stats['positive_class']]):
        ax1.text(i, v + 500, f'{v:,}', ha='center', fontweight='bold')
    
    # 2. Comparación: falla misma día vs ventana 7 días
    ax2 = axes[0, 1]
    original_failures = df['failure'].sum()
    window_positives = stats['positive_class']
    x_labels = ['Fallas reales\n(mismo día)', f'Oportunidades\nde predicción\n({window_days}d)']
    bars = ax2.bar(x_labels, [original_failures, window_positives], color=['#3498db', '#e74c3c'], edgecolor='black')
    ax2.set_title('Fallas vs Oportunidades de Detección')
    ax2.set_ylabel('Cantidad')
    for bar, val in zip(bars, [original_failures, window_positives]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, 
                f'{val:,}', ha='center', fontweight='bold')
    
    # 3. Dispositivos con/sin fallas
    ax3 = axes[1, 0]
    device_stats = stats['device_stats']
    devices_with = stats['devices_with_failure']
    devices_without = stats['total_devices'] - devices_with
    ax3.pie([devices_without, devices_with], 
            labels=[f'Sin fallas\n({devices_without})', f'Con fallas\n({devices_with})'],
            colors=['#2ecc71', '#e74c3c'], autopct='%1.1f%%', startangle=90,
            wedgeprops={'edgecolor': 'black'})
    ax3.set_title(f'Dispositivos con/sin fallas\n(Total: {stats["total_devices"]})')
    
    # 4. Distribución de días de warning por dispositivo
    ax4 = axes[1, 1]
    warning_days = device_stats[device_stats['days_before_failure'] > 0]['days_before_failure']
    if len(warning_days) > 0:
        ax4.hist(warning_days, bins=20, color='#9b59b6', edgecolor='black', alpha=0.7)
        ax4.axvline(warning_days.mean(), color='red', linestyle='--', label=f'Media: {warning_days.mean():.1f} días')
        ax4.legend()
    ax4.set_xlabel('Días con señal de warning')
    ax4.set_ylabel('Cantidad de dispositivos')
    ax4.set_title('Distribución de días de warning por dispositivo')
    
    plt.tight_layout()
    plt.show()


def plot_attribute_evolution_before_failure(df: pd.DataFrame, attributes: List[str] = None, days_before: int = 7) -> None:
    """
    Visualiza la evolución de atributos en los días previos a una falla.
    
    Args:
        df: DataFrame original
        attributes: Lista de atributos a visualizar (default: primeros 4)
        days_before: Días antes de la falla a mostrar
    """
    if attributes is None:
        attributes = ATTRIBUTES[:4]  # Solo mostrar primeros 4 para claridad
    
    pre_failure_df = analyze_pre_failure_patterns(df, days_before)
    
    if len(pre_failure_df) == 0:
        print("No hay datos suficientes de pre-falla para visualizar.")
        return
    
    # Agrupar por días hasta la falla
    grouped = pre_failure_df.groupby('days_to_failure')[attributes].agg(['mean', 'std'])
    
    # Calcular grid dinámico basado en cantidad de atributos
    n_attrs = len(attributes)
    n_cols = min(3, n_attrs)  # Máximo 3 columnas
    n_rows = (n_attrs + n_cols - 1) // n_cols  # Redondeo hacia arriba
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_attrs == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, attr in enumerate(attributes):
        ax = axes[idx]
        days = grouped.index
        means = grouped[(attr, 'mean')]
        stds = grouped[(attr, 'std')]
        
        ax.plot(days, means, 'o-', color='#e74c3c', linewidth=2, markersize=6)
        ax.fill_between(days, means - stds, means + stds, alpha=0.3, color='#e74c3c')
        ax.set_xlabel('Días hasta la falla')
        ax.set_ylabel(attr)
        ax.set_title(f'Evolución de {attr} antes de falla')
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3)
    
    # Ocultar ejes vacíos si hay más subplots que atributos
    for idx in range(n_attrs, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Evolución de Atributos en los {days_before} días previos a falla', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def print_7d_window_summary(df: pd.DataFrame, window_days: int = 7) -> None:
    """
    Imprime un resumen completo del análisis de ventana de predicción.
    
    Args:
        df: DataFrame original
        window_days: Días de la ventana de predicción
    """
    print_header(f"ANÁLISIS DE VENTANA DE PREDICCIÓN - {window_days} DÍAS", "═", 70)
    
    # Crear target y obtener estadísticas
    df_window = create_target_7d_window(df, window_days)
    stats = analyze_failure_window_distribution(df_window, window_days)
    
    print("\n📊 DEFINICIÓN DEL PROBLEMA")
    print("-" * 50)
    print(f"   Objetivo: Predecir si un dispositivo fallará")
    print(f"             en los próximos {window_days} días")
    print(f"   → True Positive (TP): Predicción=1, Falla en {window_days}d=SÍ")
    print(f"   → False Positive (FP): Predicción=1, Falla en {window_days}d=NO")
    print(f"   → True Negative (TN): Predicción=0, Falla en {window_days}d=NO")
    print(f"   → False Negative (FN): Predicción=0, Falla en {window_days}d=SÍ")
    
    print("\n📈 DISTRIBUCIÓN DE CLASES")
    print("-" * 50)
    print(f"   Total registros:      {stats['total_records']:,}")
    print(f"   Clase positiva (1):   {stats['positive_class']:,} ({stats['positive_rate']:.2f}%)")
    print(f"   Clase negativa (0):   {stats['negative_class']:,}")
    print(f"   Ratio desbalance:     {stats['imbalance_ratio']}")
    
    print("\n🔧 ANÁLISIS POR DISPOSITIVO")
    print("-" * 50)
    print(f"   Total dispositivos:              {stats['total_devices']:,}")
    print(f"   Dispositivos CON falla:          {stats['devices_with_failure']:,} ({stats['devices_with_failure']/stats['total_devices']*100:.1f}%)")
    print(f"   Dispositivos SIN falla:          {stats['total_devices'] - stats['devices_with_failure']:,}")
    print(f"   Días promedio de warning:        {stats['avg_warning_days_per_device']:.2f}")
    
    # Comparación de fallas reales vs oportunidades de predicción
    original_failures = df['failure'].sum()
    expansion_factor = stats['positive_class'] / original_failures if original_failures > 0 else 0
    
    print("\n📌 IMPACTO DE LA VENTANA DE PREDICCIÓN")
    print("-" * 50)
    print(f"   Fallas reales (día exacto):      {original_failures:,}")
    print(f"   Oportunidades de detección:      {stats['positive_class']:,}")
    print(f"   Factor de expansión:             {expansion_factor:.1f}x")
    print(f"   → Con ventana de {window_days}d, tenemos {expansion_factor:.1f}x más")
    print(f"     oportunidades para detectar una falla inminente")
    
    print("\n" + "═" * 70)


# =============================================================================
# GRÁFICOS DE PREDICCIONES
# =============================================================================
def plot_alertas_vs_fallas(
    df: pd.DataFrame,
    y_proba: np.ndarray,
    y_true: np.ndarray,
    threshold: float = 0.9,
    date_col: str = 'date'
) -> None:
    """
    Gráfico simple: ¿El modelo alertó antes de las fallas?
    
    Muestra las alertas diarias del modelo y marca los días con fallas reales.
    Si las barras naranjas aparecen ANTES de las líneas rojas, el modelo funciona.
    
    Args:
        df: DataFrame con columna de fecha
        y_proba: Probabilidades predichas por el modelo
        y_true: Valores reales (0 o 1)
        threshold: Umbral para generar alerta
        date_col: Nombre de la columna de fecha
    """
    df_plot = df.copy()
    df_plot['y_proba'] = y_proba
    df_plot['y_true'] = y_true
    df_plot['alerta'] = (y_proba >= threshold).astype(int)
    
    # Agrupar por día
    alertas_por_dia = df_plot.groupby(date_col)['alerta'].sum()
    fallas_por_dia = df_plot.groupby(date_col)['y_true'].sum()
    dias_con_falla = fallas_por_dia[fallas_por_dia > 0].index
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Barras: alertas por día
    ax.bar(alertas_por_dia.index, alertas_por_dia.values, 
           color=COLORS['warning'], alpha=0.7, width=1, label='Alertas del modelo')
    
    # Líneas verticales: días con fallas reales
    for dia in dias_con_falla:
        ax.axvline(x=dia, color=COLORS['danger'], linewidth=2, alpha=0.8)
    
    # Agregar una línea para la leyenda
    ax.axvline(x=dias_con_falla[0] if len(dias_con_falla) > 0 else alertas_por_dia.index[0], 
               color=COLORS['danger'], linewidth=2, alpha=0.8, label='Día con falla real')
    
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Cantidad de Alertas')
    ax.set_title(f'Alertas del Modelo vs Fallas Reales (threshold={threshold})')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Resumen
    total_alertas = alertas_por_dia.sum()
    total_fallas = len(dias_con_falla)
    
    resumen = f'Total: {int(total_alertas)} alertas | {total_fallas} días con fallas'
    ax.text(0.98, 0.95, resumen, transform=ax.transAxes, 
            ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# MODELO FINAL - FUNCIONES ESPECÍFICAS (Ventana 30 días)
# =============================================================================

# Configuración Final
MODEL_CONFIG = {
    'model_type': 'BalancedBaggingClassifier',
    'n_estimators': 93,
    'threshold': 0.85,  # Mejor F1 Score: 19.7%
    'window_days': 30
}

KEY_ATTRS = ['attribute2', 'attribute4', 'attribute7']


def create_features_final(df: pd.DataFrame, key_attrs: List[str] = None) -> pd.DataFrame:
    """
    Crea features para el modelo Final.
    
    Args:
        df: DataFrame con datos de dispositivos
        key_attrs: Lista de atributos clave (default: KEY_ATTRS)
    
    Returns:
        DataFrame con features creadas
    """
    if key_attrs is None:
        key_attrs = KEY_ATTRS
    
    df = df.copy()
    df = df.sort_values(['device', 'date']).reset_index(drop=True)
    
    # Rolling features
    for attr in key_attrs:
        for w in [3, 7, 14, 30]:
            df[f'{attr}_roll_mean_{w}d'] = df.groupby('device')[attr].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            )
            df[f'{attr}_roll_std_{w}d'] = df.groupby('device')[attr].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).std()
            )
        
        # Spike ratios
        for w in [7, 14, 30]:
            df[f'{attr}_spike_{w}d'] = df[attr] / (df[f'{attr}_roll_mean_{w}d'] + 1)
        
        # Diferencias
        df[f'{attr}_diff'] = df.groupby('device')[attr].diff()
        df[f'{attr}_accel'] = df.groupby('device')[f'{attr}_diff'].diff()
        
        # Z-scores
        device_mean = df.groupby('device')[attr].transform('mean')
        device_std = df.groupby('device')[attr].transform('std')
        df[f'{attr}_zscore'] = (df[attr] - device_mean) / (device_std + 1e-6)
        
        daily_mean = df.groupby('date')[attr].transform('mean')
        daily_std = df.groupby('date')[attr].transform('std')
        df[f'{attr}_fleet_zscore'] = (df[attr] - daily_mean) / (daily_std + 1e-6)
    
    # Combinaciones de spikes
    spike_cols = [f'{attr}_spike_7d' for attr in key_attrs]
    df['spike_sum'] = df[spike_cols].sum(axis=1)
    df['spike_product'] = df[spike_cols].prod(axis=1)
    df['spike_max'] = df[spike_cols].max(axis=1)
    
    # Combinaciones de z-scores
    zscore_cols = [f'{attr}_zscore' for attr in key_attrs]
    df['zscore_sum'] = df[zscore_cols].sum(axis=1)
    df['zscore_max'] = df[zscore_cols].max(axis=1)
    
    fleet_zscore_cols = [f'{attr}_fleet_zscore' for attr in key_attrs]
    df['fleet_zscore_max'] = df[fleet_zscore_cols].max(axis=1)
    
    # Reglas como features
    df['rule_spike_high'] = (
        (df['attribute7_spike_7d'] >= 5.0) &
        (df['attribute4_spike_7d'] >= 3.0) &
        (df['attribute2_spike_7d'] >= 2.0)
    ).astype(int)
    df['rule_spike_medium'] = (
        (df['attribute7_spike_7d'] >= 3.0) &
        (df['attribute4_spike_7d'] >= 2.0)
    ).astype(int)
    df['rule_spike_sum'] = (df['spike_sum'] >= 8.0).astype(int)
    df['rule_zscore'] = (df['zscore_max'] >= 3.0).astype(int)
    df['rule_combined'] = (
        (df['spike_sum'] >= 6.0) & (df['zscore_max'] >= 2.0)
    ).astype(int)
    
    rule_cols = [c for c in df.columns if c.startswith('rule_')]
    df['rules_count'] = df[rule_cols].sum(axis=1)
    df['any_rule'] = (df['rules_count'] > 0).astype(int)
    
    # Features temporales
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    
    first_date = df.groupby('device')['date'].transform('min')
    df['device_age_days'] = (df['date'] - first_date).dt.days
    
    return df.fillna(0)


def get_feature_columns_final(df: pd.DataFrame) -> List[str]:
    """Obtiene las columnas de features. (Wrapper de get_feature_columns)"""
    return get_feature_columns(df)


def evaluate_with_window_final(
    test_df: pd.DataFrame, 
    predictions: np.ndarray, 
    window_days: int = 30,
    cost_failure: float = COST_FAILURE,
    cost_maintenance: float = COST_MAINTENANCE
) -> Dict[str, Any]:
    """
    Evalúa predicciones con ventana de N días.
    
    Una predicción es TP si ocurre en los N días previos a una falla real.
    
    Args:
        test_df: DataFrame de test con columnas 'device', 'date', 'failure'
        predictions: Array de predicciones (0/1)
        window_days: Tamaño de la ventana de detección
        cost_failure: Costo de una falla no detectada
        cost_maintenance: Costo de mantenimiento preventivo
    
    Returns:
        Dict con métricas de evaluación
    """
    df = test_df.copy()
    df['prediction'] = predictions
    
    failures = df[df['failure'] == 1]
    fallas_detectadas = 0
    alertas_en_ventana = set()
    fallas_info = []
    
    for _, falla in failures.iterrows():
        device, falla_date = falla['device'], falla['date']
        window_start = falla_date - pd.Timedelta(days=window_days)
        
        mask = (df['device'] == device) & (df['date'] >= window_start) & \
               (df['date'] <= falla_date) & (df['prediction'] == 1)
        
        pred_ventana = df[mask].index.tolist()
        detectada = len(pred_ventana) > 0
        
        if detectada:
            fallas_detectadas += 1
            alertas_en_ventana.update(pred_ventana)
        
        fallas_info.append({
            'device': device,
            'falla_date': falla_date,
            'detectada': detectada,
            'alertas_ventana': len(pred_ventana)
        })
    
    todas_pred = set(df[df['prediction'] == 1].index.tolist())
    fp = len(todas_pred - alertas_en_ventana)
    tp, fn = fallas_detectadas, len(failures) - fallas_detectadas
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    cost = tp * cost_maintenance + fp * cost_maintenance + fn * cost_failure
    baseline = len(failures) * cost_failure
    savings_pct = (baseline - cost) / baseline * 100 if baseline > 0 else 0
    
    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': precision, 'recall': recall, 'f1': f1,
        'cost': cost, 'baseline': baseline, 'savings_pct': savings_pct,
        'fallas_info': pd.DataFrame(fallas_info)
    }


def load_data_final(data_path: str) -> pd.DataFrame:
    """Carga y prepara los datos para el modelo Final. (Wrapper de load_data)"""
    return load_data(data_path, sort_by_date=True)


def temporal_split_final(
    df: pd.DataFrame, 
    train_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Realiza split temporal de los datos. (Wrapper de temporal_split)"""
    return temporal_split(df, train_ratio)


def train_model_final(
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    n_estimators: int = None,
    random_state: int = 42
):
    """
    Entrena el modelo Final (BalancedBaggingClassifier).
    
    Returns:
        Modelo entrenado
    """
    from imblearn.ensemble import BalancedBaggingClassifier
    
    if n_estimators is None:
        n_estimators = MODEL_CONFIG['n_estimators']
    
    model = BalancedBaggingClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    return model


def predict_final(
    model, 
    X: pd.DataFrame, 
    threshold: float = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Genera predicciones con el modelo Final. (Wrapper de predict_with_threshold)"""
    if threshold is None:
        threshold = MODEL_CONFIG['threshold']
    return predict_with_threshold(model, X, threshold)


def analyze_thresholds_final(
    test_df: pd.DataFrame,
    y_proba: np.ndarray,
    window_days: int = 30,
    threshold_step: float = 0.05
) -> pd.DataFrame:
    """
    Analiza métricas para diferentes thresholds.
    
    Args:
        test_df: DataFrame de test
        y_proba: Probabilidades predichas
        window_days: Ventana de evaluación
        threshold_step: Paso entre thresholds
    
    Returns:
        DataFrame con métricas por threshold
    """
    thresholds = np.arange(threshold_step, 1.0, threshold_step)
    results = []
    
    for thresh in thresholds:
        y_pred_t = (y_proba >= thresh).astype(int)
        m = evaluate_with_window_final(test_df, y_pred_t, window_days)
        
        total_predictions = len(y_pred_t)
        tn = total_predictions - m['tp'] - m['fp'] - m['fn']
        
        results.append({
            'threshold': thresh,
            'tp': m['tp'],
            'fp': m['fp'],
            'fn': m['fn'],
            'tn': tn,
            'precision': m['precision'],
            'recall': m['recall'],
            'f1': m['f1'],
            'cost': m['cost'],
            'savings_pct': m['savings_pct']
        })
    
    return pd.DataFrame(results)


def display_threshold_table_final(df_thresh: pd.DataFrame, current_threshold: float = None) -> None:
    """Muestra tabla formateada de métricas por threshold. (Wrapper de display_threshold_table)"""
    display_threshold_table(df_thresh, current_threshold)


def plot_probability_distribution_final(
    y_proba: np.ndarray, 
    y_true: np.ndarray, 
    threshold: float = None
) -> None:
    """Visualiza distribución de probabilidades por clase."""
    if threshold is None:
        threshold = MODEL_CONFIG['threshold']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histograma
    ax1 = axes[0]
    ax1.hist(y_proba[y_true == 0], bins=50, alpha=0.7, label='No Falla', 
             color=COLORS['success'], density=True)
    ax1.hist(y_proba[y_true == 1], bins=50, alpha=0.7, label='Falla', 
             color=COLORS['danger'], density=True)
    ax1.axvline(x=threshold, color=COLORS['dark'], linestyle='--', linewidth=2, 
                label=f'Threshold = {threshold}')
    ax1.set_xlabel('Probabilidad de Falla')
    ax1.set_ylabel('Densidad')
    ax1.set_title('Distribución de Probabilidades por Clase')
    ax1.legend()
    
    # Box plot
    ax2 = axes[1]
    data_box = [y_proba[y_true == 0], y_proba[y_true == 1]]
    bp = ax2.boxplot(data_box, labels=['No Falla', 'Falla'], patch_artist=True)
    bp['boxes'][0].set_facecolor(COLORS['success'])
    bp['boxes'][1].set_facecolor(COLORS['danger'])
    ax2.axhline(y=threshold, color=COLORS['dark'], linestyle='--', linewidth=2,
                label=f'Threshold = {threshold}')
    ax2.set_ylabel('Probabilidad de Falla')
    ax2.set_title('Distribución de Probabilidades (Box Plot)')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


def plot_predictions_over_time_final(
    df: pd.DataFrame, 
    y_proba: np.ndarray, 
    y_true: np.ndarray, 
    threshold: float = None
) -> None:
    """
    Vista simplificada tipo semáforo: cada día un color según resultado.
    
    Colores:
    - Verde: Falla detectada correctamente (TP)
    - Rojo: Falla NO detectada (FN) 
    - Amarillo: Falsa alarma (FP)
    - Gris: Día normal sin eventos (TN)
    """
    from matplotlib.patches import Patch
    
    if threshold is None:
        threshold = MODEL_CONFIG['threshold']
    
    df_plot = df.copy()
    df_plot['y_proba'] = y_proba
    df_plot['y_true'] = y_true
    df_plot['y_pred'] = (y_proba >= threshold).astype(int)
    
    # Clasificar cada observación
    df_plot['tp'] = ((df_plot['y_true'] == 1) & (df_plot['y_pred'] == 1)).astype(int)
    df_plot['fn'] = ((df_plot['y_true'] == 1) & (df_plot['y_pred'] == 0)).astype(int)
    df_plot['fp'] = ((df_plot['y_true'] == 0) & (df_plot['y_pred'] == 1)).astype(int)
    
    # Resumen diario
    daily = df_plot.groupby('date').agg({
        'y_true': 'sum',
        'y_pred': 'sum',
        'tp': 'sum',
        'fn': 'sum',
        'fp': 'sum'
    })
    
    # Colorear cada día según prioridad: FN > TP > FP > TN
    colors = []
    for _, row in daily.iterrows():
        if row['fn'] > 0:
            colors.append('#e74c3c')  # Rojo - falla no detectada (crítico)
        elif row['tp'] > 0:
            colors.append('#27ae60')  # Verde - falla detectada
        elif row['fp'] > 0:
            colors.append('#f39c12')  # Amarillo - falsa alarma
        else:
            colors.append('#ecf0f1')  # Gris - día normal
    
    # Calcular métricas
    total_fallas = df_plot['y_true'].sum()
    fallas_detectadas = df_plot['tp'].sum()
    fallas_perdidas = df_plot['fn'].sum()
    falsas_alarmas = df_plot['fp'].sum()
    recall = fallas_detectadas / total_fallas * 100 if total_fallas > 0 else 0
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(14, 4))
    
    # Barras de colores por día
    ax.bar(daily.index, [1]*len(daily), color=colors, width=1, edgecolor='white', linewidth=0.5)
    
    # Leyenda
    legend_elements = [
        Patch(facecolor='#27ae60', edgecolor='white', label=f'Falla Detectada ({fallas_detectadas})'),
        Patch(facecolor='#e74c3c', edgecolor='white', label=f'Falla NO Detectada ({fallas_perdidas})'),
        Patch(facecolor='#f39c12', edgecolor='white', label=f'Falsa Alarma ({falsas_alarmas})'),
        Patch(facecolor='#ecf0f1', edgecolor='gray', label='Día Normal')
    ]
    ax.legend(handles=legend_elements, loc='upper right', ncol=4, fontsize=9)
    
    # Formato
    ax.set_yticks([])
    ax.set_xlabel('Fecha')
    ax.set_title(f'Resumen de Predicciones por Día | Recall: {recall:.1f}% ({fallas_detectadas}/{total_fallas} fallas) | Threshold: {threshold}', 
                 fontsize=12, fontweight='bold')
    
    # Rotar fechas si hay muchas
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()


def plot_cost_comparison_final(
    df: pd.DataFrame, 
    y_pred: np.ndarray, 
    y_true: np.ndarray,
    model_name: str = 'Final'
) -> None:
    """Visualiza comparación de costos modelo vs baseline."""
    df_cost = df.copy()
    df_cost['y_pred'] = y_pred
    df_cost['y_true'] = y_true
    
    df_cost['cost_baseline'] = df_cost['y_true'] * COST_FAILURE
    df_cost['cost_model'] = df_cost['y_pred'] * COST_MAINTENANCE
    
    daily_baseline = df_cost.groupby('date')['cost_baseline'].sum().cumsum()
    daily_model_pred = df_cost.groupby('date')['cost_model'].sum().cumsum()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Panel 1: Costo acumulado
    ax1 = axes[0]
    ax1.plot(daily_baseline.index, daily_baseline.values, 
             color=COLORS['danger'], linewidth=2, label='Baseline (sin modelo)')
    ax1.plot(daily_model_pred.index, daily_model_pred.values, 
             color=COLORS['success'], linewidth=2, label=f'Costo Mantenimientos ({model_name})')
    ax1.fill_between(daily_baseline.index, daily_baseline.values, daily_model_pred.values,
                     alpha=0.3, color=COLORS['success'])
    ax1.set_ylabel('Costo Acumulado ($)')
    ax1.set_title(f'Costo Acumulado: Baseline vs {model_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Diferencia
    ax2 = axes[1]
    diff = daily_baseline - daily_model_pred
    colors_bar = [COLORS['success'] if d > 0 else COLORS['danger'] for d in diff.values]
    ax2.bar(diff.index, diff.values, color=colors_bar, alpha=0.7, width=1)
    ax2.axhline(y=0, color='black', linewidth=1)
    ax2.set_ylabel('Diferencia de Costo ($)')
    ax2.set_xlabel('Fecha')
    ax2.set_title('Diferencia: Baseline - Modelo (verde = ahorro)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_cost_comparison_multi_threshold_final(
    df: pd.DataFrame, 
    y_proba: np.ndarray, 
    y_true: np.ndarray,
    thresholds: List[float] = [0.85, 0.90],
    model_name: str = 'Final'
) -> None:
    """
    Visualiza comparación de costos entre baseline y múltiples thresholds.
    
    Args:
        df: DataFrame con columna 'date'
        y_proba: Probabilidades predichas
        y_true: Valores reales
        thresholds: Lista de thresholds a comparar
        model_name: Nombre del modelo
    """
    df_cost = df.copy()
    df_cost['y_true'] = y_true
    
    # Calcular costo baseline
    df_cost['cost_baseline'] = df_cost['y_true'] * COST_FAILURE
    daily_baseline = df_cost.groupby('date')['cost_baseline'].sum().cumsum()
    
    # Calcular costos para cada threshold
    daily_models = {}
    total_costs = {}
    for th in thresholds:
        y_pred_th = (y_proba >= th).astype(int)
        df_cost[f'cost_model_{th}'] = y_pred_th * COST_MAINTENANCE
        daily_models[th] = df_cost.groupby('date')[f'cost_model_{th}'].sum().cumsum()
        total_costs[th] = df_cost[f'cost_model_{th}'].sum()
    
    # Colores para los diferentes thresholds
    colors_th = ['#27ae60', '#3498db', '#9b59b6', '#f39c12', '#1abc9c']
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Panel 1: Costo acumulado
    ax1 = axes[0]
    ax1.plot(daily_baseline.index, daily_baseline.values, 
             color=COLORS['danger'], linewidth=2.5, label='Baseline (sin modelo)', linestyle='--')
    
    for i, th in enumerate(thresholds):
        color = colors_th[i % len(colors_th)]
        ax1.plot(daily_models[th].index, daily_models[th].values, 
                 color=color, linewidth=2, label=f'{model_name} (threshold={th})')
    
    ax1.set_ylabel('Costo Acumulado ($)')
    ax1.set_title(f'Costo Acumulado: Baseline vs {model_name} con Diferentes Thresholds')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Barras comparativas
    ax2 = axes[1]
    total_baseline = df_cost['cost_baseline'].sum()
    
    labels = ['Baseline\n(sin modelo)'] + [f'{model_name}\n(th={th})' for th in thresholds]
    values = [total_baseline] + [total_costs[th] for th in thresholds]
    bar_colors = [COLORS['danger']] + [colors_th[i % len(colors_th)] for i in range(len(thresholds))]
    
    bars = ax2.bar(labels, values, color=bar_colors)
    ax2.set_ylabel('Costo Total ($)')
    ax2.set_title('Comparación de Costo Total por Threshold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores en las barras
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total_baseline*0.01, 
                 f'${val:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Resumen
    print(f"\n{'='*60}")
    print("COMPARACIÓN DE COSTOS POR THRESHOLD")
    print(f"{'='*60}")
    print(f"  Costo Baseline (sin modelo): ${total_baseline:.2f}")
    for th in thresholds:
        ahorro = total_baseline - total_costs[th]
        ahorro_pct = (ahorro / total_baseline) * 100 if total_baseline > 0 else 0
        print(f"  Costo con threshold={th}: ${total_costs[th]:.2f} (Ahorro: ${ahorro:.2f}, {ahorro_pct:.1f}%)")
    print(f"{'='*60}")


def plot_cost_comparison_normalized_final(
    df: pd.DataFrame, 
    y_pred: np.ndarray, 
    y_true: np.ndarray,
    cost_failure: float = 1.0,
    cost_maintenance: float = 0.1,
    model_name: str = 'Final',
    threshold: float = None
) -> None:
    """
    Visualiza comparación de costos modelo vs baseline con costos personalizables.
    
    Args:
        df: DataFrame con columna 'date'
        y_pred: Predicciones binarias
        y_true: Valores reales
        cost_failure: Costo por fallo no atendido (default: 1.0)
        cost_maintenance: Costo por mantenimiento preventivo (default: 0.1)
        model_name: Nombre del modelo para el título
        threshold: Threshold usado (opcional, para mostrar en título)
    """
    df_cost = df.copy()
    df_cost['y_pred'] = y_pred
    df_cost['y_true'] = y_true
    
    # Calcular costos
    df_cost['cost_baseline'] = df_cost['y_true'] * cost_failure
    df_cost['cost_model'] = df_cost['y_pred'] * cost_maintenance
    
    # Agregar costo de fallos no detectados (FN)
    df_cost['cost_fn'] = ((df_cost['y_true'] == 1) & (df_cost['y_pred'] == 0)).astype(int) * cost_failure
    df_cost['cost_model_total'] = df_cost['cost_model'] + df_cost['cost_fn']
    
    # Acumulados diarios
    daily_baseline = df_cost.groupby('date')['cost_baseline'].sum().cumsum()
    daily_model = df_cost.groupby('date')['cost_model_total'].sum().cumsum()
    
    # Visualización
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Panel 1: Costo acumulado
    ax1 = axes[0]
    ax1.plot(daily_baseline.index, daily_baseline.values, 
             color=COLORS['danger'], linewidth=2, 
             label=f'Baseline (sin modelo) - Costo={cost_failure} por fallo')
    
    th_str = f' (threshold={threshold})' if threshold else ''
    ax1.plot(daily_model.index, daily_model.values, 
             color=COLORS['success'], linewidth=2, 
             label=f'{model_name}{th_str} - Mant={cost_maintenance}')
    
    ax1.fill_between(daily_baseline.index, daily_baseline.values, daily_model.values,
                     alpha=0.3, color=COLORS['success'], 
                     where=daily_baseline.values >= daily_model.values)
    ax1.fill_between(daily_baseline.index, daily_baseline.values, daily_model.values,
                     alpha=0.3, color=COLORS['danger'], 
                     where=daily_baseline.values < daily_model.values)
    
    ax1.set_ylabel('Costo Acumulado (unidades)')
    ax1.set_title(f'Comparación de Costos Normalizado: Baseline vs {model_name}\n'
                  f'(Costo Fallo={cost_failure}, Costo Mantenimiento={cost_maintenance})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Resumen en barras
    ax2 = axes[1]
    total_baseline = df_cost['cost_baseline'].sum()
    total_model = df_cost['cost_model_total'].sum()
    ahorro = total_baseline - total_model
    ahorro_pct = (ahorro / total_baseline) * 100 if total_baseline > 0 else 0
    
    bars = ax2.bar(
        ['Baseline\n(sin modelo)', f'{model_name}\n(th={threshold})' if threshold else model_name], 
        [total_baseline, total_model], 
        color=[COLORS['danger'], COLORS['success']]
    )
    ax2.set_ylabel('Costo Total (unidades)')
    ax2.set_title(f'Costo Total: Ahorro = {ahorro:.2f} unidades ({ahorro_pct:.1f}%)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores en las barras
    for bar, val in zip(bars, [total_baseline, total_model]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                 f'{val:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Resumen detallado
    print(f"\n{'='*50}")
    print("RESUMEN DE COSTOS NORMALIZADO")
    print(f"{'='*50}")
    print(f"  Costo por fallo no atendido: {cost_failure}")
    print(f"  Costo por mantenimiento preventivo: {cost_maintenance}")
    print(f"{'='*50}")
    print(f"  Total fallos reales: {int(df_cost['y_true'].sum())}")
    print(f"  Total mantenimientos predichos: {int(df_cost['y_pred'].sum())}")
    print(f"  Fallos detectados (TP): {((df_cost['y_true'] == 1) & (df_cost['y_pred'] == 1)).sum()}")
    print(f"  Fallos no detectados (FN): {((df_cost['y_true'] == 1) & (df_cost['y_pred'] == 0)).sum()}")
    print(f"{'='*50}")
    print(f"  Costo Baseline (sin modelo): {total_baseline:.2f}")
    print(f"  Costo con Modelo: {total_model:.2f}")
    print(f"  AHORRO: {ahorro:.2f} ({ahorro_pct:.1f}%)")
    print(f"{'='*50}")


def plot_confusion_matrix_final(metrics: Dict[str, Any], window_days: int = 30) -> None:
    """Visualiza matriz de confusión con costos. (Wrapper de plot_confusion_matrix_with_costs)"""
    plot_confusion_matrix_with_costs(metrics, window_days)


def plot_feature_importance_final(
    model, 
    feature_cols: List[str], 
    top_n: int = 20,
    X_val: pd.DataFrame = None,
    y_val: np.ndarray = None,
    use_permutation: bool = False
) -> None:
    """
    Visualiza importancia de features.
    
    Args:
        model: Modelo entrenado
        feature_cols: Nombres de features
        top_n: Número de features a mostrar
        X_val: Datos de validación (opcional, para permutation importance)
        y_val: Labels de validación (opcional, para permutation importance)
        use_permutation: Si True, usa permutation importance (más preciso pero lento)
    """
    if use_permutation and X_val is not None and y_val is not None:
        plot_permutation_importance(model, X_val, y_val, feature_cols, top_n)
    else:
        plot_feature_importance_generic(model, feature_cols, top_n, highlight_pattern='rule')


def plot_permutation_importance(
    model,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    feature_cols: List[str],
    top_n: int = 20
) -> None:
    """
    Visualiza importancia de features usando permutation importance.
    Más preciso para modelos ensemble como BalancedBaggingClassifier.
    
    Args:
        model: Modelo entrenado
        X_val: Datos de validación
        y_val: Labels de validación
        feature_cols: Nombres de features
        top_n: Número de features a mostrar
    """
    from sklearn.inspection import permutation_importance
    
    print("Calculando permutation importance (esto puede tardar unos segundos)...")
    
    result = permutation_importance(
        model, X_val, y_val, 
        n_repeats=10, 
        random_state=42, 
        n_jobs=-1
    )
    
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': result.importances_mean,
        'std': result.importances_std
    }).sort_values('importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    top_features = importance_df.tail(top_n)
    colors_imp = [COLORS['primary'] if 'rule' not in f else COLORS['warning'] 
                  for f in top_features['feature']]
    
    ax.barh(top_features['feature'], top_features['importance'], 
            xerr=top_features['std'], color=colors_imp, capsize=3)
    ax.set_xlabel('Importancia (reducción en accuracy)')
    ax.set_title(f'Top {top_n} Features más Importantes (Permutation Importance)')
    
    plt.tight_layout()
    plt.show()


def plot_threshold_analysis_final(
    df_thresh: pd.DataFrame, 
    metrics: Dict[str, Any],
    current_threshold: float = None
) -> None:
    """Visualiza análisis de thresholds. (Wrapper de plot_threshold_analysis_generic)"""
    if current_threshold is None:
        current_threshold = MODEL_CONFIG['threshold']
    plot_threshold_analysis_generic(df_thresh, metrics['baseline'], current_threshold, metrics['cost'])


def plot_failures_detection_final(fallas_df: pd.DataFrame) -> None:
    """Visualiza fallas detectadas vs no detectadas."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    detectadas = fallas_df[fallas_df['detectada']]
    no_detectadas = fallas_df[~fallas_df['detectada']]
    
    ax.scatter(detectadas['falla_date'], [1]*len(detectadas), 
               color=COLORS['success'], s=200, marker='o', 
               label=f'Detectadas ({len(detectadas)})')
    ax.scatter(no_detectadas['falla_date'], [0]*len(no_detectadas), 
               color=COLORS['danger'], s=200, marker='X', 
               label=f'No detectadas ({len(no_detectadas)})')
    
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['No Detectada', 'Detectada'])
    ax.set_xlabel('Fecha de Falla')
    ax.set_title(f'Fallas en Período de Test: {len(detectadas)} detectadas de {len(fallas_df)} totales')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_device_probability_evolution(
    df: pd.DataFrame,
    y_proba: np.ndarray,
    fallas_info: pd.DataFrame,
    threshold: float = None,
    window_days: int = 30,
    n_examples: int = 4,
    only_detected: bool = True
) -> None:
    """
    Visualiza la evolución de probabilidades de falla para dispositivos específicos.
    
    Muestra cómo el modelo detecta el aumento de probabilidad antes de la falla
    y cuándo se cruza el threshold para decidir mantenimiento.
    
    Args:
        df: DataFrame con datos de test (debe tener 'device', 'date')
        y_proba: Probabilidades predichas
        fallas_info: DataFrame con info de fallas (de evaluate_with_window_final)
        threshold: Umbral de decisión
        window_days: Ventana de detección
        n_examples: Número de ejemplos a mostrar
        only_detected: Si True, solo muestra fallas detectadas correctamente
    """
    if threshold is None:
        threshold = MODEL_CONFIG['threshold']
    
    # Preparar datos
    df_plot = df.copy()
    df_plot['y_proba'] = y_proba
    
    # Filtrar fallas según criterio
    if only_detected:
        fallas_mostrar = fallas_info[fallas_info['detectada'] == True].head(n_examples)
        titulo_tipo = "DETECTADAS CORRECTAMENTE"
    else:
        fallas_mostrar = fallas_info[fallas_info['detectada'] == False].head(n_examples)
        titulo_tipo = "NO DETECTADAS"
    
    if len(fallas_mostrar) == 0:
        print(f"No hay fallas {'detectadas' if only_detected else 'no detectadas'} para mostrar.")
        return
    
    # Crear subplots
    n_plots = min(len(fallas_mostrar), n_examples)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots))
    if n_plots == 1:
        axes = [axes]
    
    fig.suptitle(f'EVOLUCIÓN DE PROBABILIDAD DE FALLA - CASOS {titulo_tipo}', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    for idx, (_, falla) in enumerate(fallas_mostrar.iterrows()):
        ax = axes[idx]
        device = falla['device']
        falla_date = falla['falla_date']
        
        # Filtrar datos del dispositivo
        device_data = df_plot[df_plot['device'] == device].sort_values('date')
        
        if len(device_data) == 0:
            continue
        
        # Calcular ventana
        window_start = falla_date - pd.Timedelta(days=window_days)
        
        # Plot de probabilidad
        ax.plot(device_data['date'], device_data['y_proba'], 
                color=COLORS['primary'], linewidth=2, label='Probabilidad de falla')
        
        # Línea de threshold
        ax.axhline(y=threshold, color=COLORS['dark'], linestyle='--', linewidth=2,
                   label=f'Threshold = {threshold}')
        
        # Marcar la falla real
        ax.axvline(x=falla_date, color=COLORS['danger'], linewidth=3, 
                   label='Falla real', alpha=0.8)
        
        # Sombrear ventana de detección
        ax.axvspan(window_start, falla_date, alpha=0.2, color=COLORS['warning'],
                   label=f'Ventana {window_days}d')
        
        # Marcar puntos donde se cruza el threshold (decisión de mantenimiento)
        alertas = device_data[device_data['y_proba'] >= threshold]
        alertas_en_ventana = alertas[(alertas['date'] >= window_start) & (alertas['date'] <= falla_date)]
        
        if len(alertas_en_ventana) > 0:
            primera_alerta = alertas_en_ventana.iloc[0]
            ax.scatter([primera_alerta['date']], [primera_alerta['y_proba']], 
                      color=COLORS['success'], s=200, marker='*', zorder=5,
                      label='Primera alerta en ventana')
            
            # Calcular días de anticipación
            dias_anticipacion = (falla_date - primera_alerta['date']).days
            ax.annotate(f'Alerta {dias_anticipacion}d antes', 
                       xy=(primera_alerta['date'], primera_alerta['y_proba']),
                       xytext=(10, 20), textcoords='offset points',
                       fontsize=10, fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color=COLORS['success']),
                       color=COLORS['success'])
        
        # Configurar ejes
        ax.set_xlim(device_data['date'].min(), device_data['date'].max())
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Probabilidad')
        ax.set_title(f'Dispositivo: {device} | Falla: {falla_date.strftime("%Y-%m-%d")}', 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Añadir zona de "decisión de mantenimiento"
        ax.fill_between(device_data['date'], threshold, 1, 
                       where=(device_data['y_proba'] >= threshold),
                       alpha=0.3, color=COLORS['success'], 
                       label='Zona de mantenimiento')
    
    plt.tight_layout()
    plt.show()


def plot_detection_timeline(
    df: pd.DataFrame,
    y_proba: np.ndarray,
    fallas_info: pd.DataFrame,
    threshold: float = None,
    window_days: int = 30
) -> None:
    """
    Muestra un timeline resumido de todas las detecciones exitosas.
    
    Args:
        df: DataFrame con datos de test
        y_proba: Probabilidades predichas
        fallas_info: DataFrame con info de fallas
        threshold: Umbral de decisión
        window_days: Ventana de detección
    """
    if threshold is None:
        threshold = MODEL_CONFIG['threshold']
    
    df_plot = df.copy()
    df_plot['y_proba'] = y_proba
    
    # Solo fallas detectadas
    detectadas = fallas_info[fallas_info['detectada'] == True]
    
    if len(detectadas) == 0:
        print("No hay fallas detectadas para mostrar.")
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    anticipaciones = []
    
    for i, (_, falla) in enumerate(detectadas.iterrows()):
        device = falla['device']
        falla_date = falla['falla_date']
        window_start = falla_date - pd.Timedelta(days=window_days)
        
        # Datos del dispositivo
        device_data = df_plot[df_plot['device'] == device]
        alertas = device_data[(device_data['y_proba'] >= threshold) & 
                              (device_data['date'] >= window_start) & 
                              (device_data['date'] <= falla_date)]
        
        if len(alertas) > 0:
            primera_alerta = alertas['date'].min()
            dias_anticipacion = (falla_date - primera_alerta).days
            anticipaciones.append(dias_anticipacion)
            
            # Barra horizontal para cada falla
            ax.barh(i, dias_anticipacion, color=COLORS['success'], alpha=0.7, height=0.6)
            ax.scatter([dias_anticipacion], [i], color=COLORS['danger'], s=100, 
                      marker='X', zorder=5, label='Falla' if i == 0 else '')
            ax.text(dias_anticipacion + 0.5, i, f'{device}', va='center', fontsize=9)
    
    ax.set_xlabel('Días de anticipación antes de la falla')
    ax.set_ylabel('Fallas detectadas')
    ax.set_title(f'Tiempo de Anticipación en Detecciones Exitosas\n'
                f'Promedio: {np.mean(anticipaciones):.1f} días | '
                f'Mínimo: {np.min(anticipaciones)} días | '
                f'Máximo: {np.max(anticipaciones)} días')
    ax.axvline(x=np.mean(anticipaciones), color=COLORS['primary'], linestyle='--', 
               linewidth=2, label=f'Promedio ({np.mean(anticipaciones):.1f}d)')
    ax.set_xlim(0, window_days + 2)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()
    
    # Estadísticas
    print("\n" + "="*60)
    print(" ESTADÍSTICAS DE ANTICIPACIÓN")
    print("="*60)
    print(f"  Fallas detectadas: {len(detectadas)}")
    print(f"  Días de anticipación promedio: {np.mean(anticipaciones):.1f}")
    print(f"  Días de anticipación mínimo: {np.min(anticipaciones)}")
    print(f"  Días de anticipación máximo: {np.max(anticipaciones)}")
    print(f"  Mediana: {np.median(anticipaciones):.1f} días")
    print("="*60)
    
    plt.tight_layout()
    plt.show()


def print_results_final(
    metrics: Dict[str, Any], 
    config: Dict[str, Any] = None,
    window_days: int = 30
) -> None:
    """Imprime resultados del modelo Final."""
    if config is None:
        config = MODEL_CONFIG
    
    print("=" * 60)
    print(" RESULTADOS MODELO FINAL")
    print("=" * 60)
    print(f"\n  Threshold: {config['threshold']}")
    print(f"  Ventana: {window_days} días")
    print(f"\n  True Positives (fallas detectadas): {metrics['tp']}")
    print(f"  False Positives (alertas falsas): {metrics['fp']}")
    print(f"  False Negatives (fallas no detectadas): {metrics['fn']}")
    print(f"\n  Precision: {metrics['precision']*100:.1f}%")
    print(f"  Recall: {metrics['recall']*100:.1f}%")
    print(f"  F1 Score: {metrics['f1']*100:.1f}%")
    print(f"\n  BASELINE: ${metrics['baseline']:.1f}")
    print(f"  COSTO MODELO: ${metrics['cost']:.1f}")
    print(f"  AHORRO: {metrics['savings_pct']:+.1f}%")
    print("=" * 60)


def print_summary_final(
    config: Dict[str, Any],
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: List[str],
    metrics: Dict[str, Any],
    window_days: int = 30
) -> None:
    """Imprime resumen ejecutivo del modelo Final."""
    print("\n" + "=" * 70)
    print(" RESUMEN MODELO FINAL - MANTENIMIENTO PREDICTIVO")
    print("=" * 70)
    
    print(f"""
CONFIGURACIÓN:
  Modelo: {config['model_type']}
  n_estimators: {config['n_estimators']}
  Threshold: {config['threshold']}
  Ventana de detección: {window_days} días

DATASET:
  Train: {len(df_train):,} registros ({df_train['failure'].sum()} fallas)
  Test: {len(df_test):,} registros ({df_test['failure'].sum()} fallas)
  Features: {len(feature_cols)}

RESULTADOS:
  True Positives: {metrics['tp']} fallas detectadas
  False Positives: {metrics['fp']} alertas falsas
  False Negatives: {metrics['fn']} fallas no detectadas

  Precision: {metrics['precision']*100:.1f}%
  Recall: {metrics['recall']*100:.1f}%
  F1 Score: {metrics['f1']*100:.1f}%

COSTOS:
  Baseline (sin modelo): ${metrics['baseline']:.1f}
  Costo con modelo: ${metrics['cost']:.1f}
  AHORRO: {metrics['savings_pct']:+.1f}% {'✓' if metrics['savings_pct'] > 0 else ''}
""")
    
    print("=" * 70)


def save_model_final(
    model, 
    feature_cols: List[str], 
    metrics: Dict[str, Any],
    config: Dict[str, Any] = None,
    filepath: str = 'modelo_final_final.pkl'
) -> None:
    """Guarda el modelo Final y sus artefactos."""
    if config is None:
        config = MODEL_CONFIG
    
    model_data = {
        'model': model,
        'feature_cols': feature_cols,
        'threshold': config['threshold'],
        'config': config,
        'metrics': metrics
    }
    
    joblib.dump(model_data, filepath)
    print(f"Modelo guardado en: {filepath}")


def plot_final_summary_final(
    metrics: Dict[str, Any],
    model_name: str = "Modelo Final",
    window_days: int = 30
) -> None:
    """
    Visualiza resumen final del modelo con métricas, costos y detección de fallas.
    
    Args:
        metrics: Diccionario con métricas (precision, recall, f1, tp, fp, fn, baseline, cost, savings_pct)
        model_name: Nombre del modelo para mostrar
        window_days: Ventana de detección en días
    """
    fig = plt.figure(figsize=(16, 10))
    
    # Calcular valores
    total_fallas = metrics['tp'] + metrics['fn']
    ahorro = metrics['baseline'] - metrics['cost']
    
    # --- Panel 1: Métricas de Clasificación ---
    ax1 = fig.add_subplot(2, 2, 1)
    metricas_nombres = ['Precision', 'Recall', 'F1 Score']
    valores = [metrics['precision']*100, metrics['recall']*100, metrics['f1']*100]
    colores = [COLORS['primary'], COLORS['warning'], COLORS['success']]
    
    bars = ax1.bar(metricas_nombres, valores, color=colores, edgecolor='black', linewidth=1.5)
    ax1.set_ylim(0, 110)
    ax1.set_ylabel('Porcentaje (%)', fontsize=12)
    ax1.set_title('MÉTRICAS DE CLASIFICACIÓN', fontsize=14, fontweight='bold', pad=15)
    
    for bar, val in zip(bars, valores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 3, 
                 f'{val:.1f}%', ha='center', fontsize=13, fontweight='bold')
    
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax1.text(2.4, 52, 'Baseline 50%', fontsize=9, color='gray', va='bottom')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_axisbelow(True)
    
    # --- Panel 2: Comparación de Costos ---
    ax2 = fig.add_subplot(2, 2, 2)
    costos_nombres = ['Sin Modelo\n(Baseline)', f'Con {model_name}']
    valores_costo = [metrics['baseline'], metrics['cost']]
    colores_costo = [COLORS['danger'], COLORS['success']]
    
    bars2 = ax2.bar(costos_nombres, valores_costo, color=colores_costo, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Costo ($)', fontsize=12)
    ax2.set_title('COMPARACIÓN DE COSTOS', fontsize=14, fontweight='bold', pad=15)
    
    max_costo = max(valores_costo) * 1.25
    ax2.set_ylim(0, max_costo)
    
    for bar, val in zip(bars2, valores_costo):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_costo*0.02, 
                 f'${val:.1f}', ha='center', fontsize=13, fontweight='bold')
    
    # Mostrar ahorro solo si es positivo
    if ahorro > 0:
        mid_y = (metrics['baseline'] + metrics['cost']) / 2
        ax2.annotate('', xy=(1, metrics['cost'] + max_costo*0.02), 
                     xytext=(0, metrics['baseline'] - max_costo*0.02),
                     arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=2.5, 
                                     connectionstyle='arc3,rad=0.2'))
        ax2.text(0.5, mid_y, f'Ahorro\n${ahorro:.1f}\n({metrics["savings_pct"]:+.1f}%)',
                ha='center', fontsize=11, fontweight='bold', color=COLORS['success'],
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLORS['success'], alpha=0.9))
    elif ahorro == 0:
        ax2.text(0.5, max_costo*0.5, 'Sin ahorro\n(igual costo)',
                ha='center', fontsize=11, fontweight='bold', color=COLORS['dark'],
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor=COLORS['warning'], alpha=0.9))
    
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_axisbelow(True)
    
    # --- Panel 3: Detección de Fallas (Barras horizontales) ---
    ax3 = fig.add_subplot(2, 2, 3)
    
    categorias = ['Fallas Detectadas (TP)', 'Fallas No Detectadas (FN)', 'Falsas Alarmas (FP)']
    valores_det = [metrics['tp'], metrics['fn'], metrics['fp']]
    colores_det = [COLORS['success'], COLORS['danger'], COLORS['warning']]
    
    y_pos = np.arange(len(categorias))
    bars3 = ax3.barh(y_pos, valores_det, color=colores_det, edgecolor='black', linewidth=1.5, height=0.6)
    
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(categorias, fontsize=11)
    ax3.set_xlabel('Cantidad', fontsize=12)
    ax3.set_title('DETECCIÓN DE FALLAS', fontsize=14, fontweight='bold', pad=15)
    ax3.invert_yaxis()
    
    # Valores en las barras
    for bar, val in zip(bars3, valores_det):
        width = bar.get_width()
        ax3.text(width + 0.3, bar.get_y() + bar.get_height()/2, 
                 str(int(val)), ha='left', va='center', fontsize=13, fontweight='bold')
    
    # Agregar info de total
    ax3.text(0.98, 0.02, f'Total fallas reales: {total_fallas}', 
             transform=ax3.transAxes, ha='right', va='bottom',
             fontsize=10, style='italic', color=COLORS['dark'])
    
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.set_axisbelow(True)
    max_val = max(valores_det) * 1.3 if max(valores_det) > 0 else 5
    ax3.set_xlim(0, max_val)
    
    # --- Panel 4: Resumen Ejecutivo ---
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    # Construir resumen
    recall_pct = metrics['recall'] * 100
    precision_pct = metrics['precision'] * 100
    f1_pct = metrics['f1'] * 100
    
    # Determinar estado del modelo
    if recall_pct >= 50:
        estado = "ACEPTABLE"
        estado_color = COLORS['success']
    elif recall_pct >= 20:
        estado = "MEJORABLE"
        estado_color = COLORS['warning']
    else:
        estado = "INSUFICIENTE"
        estado_color = COLORS['danger']
    
    # Crear texto con formato limpio
    resumen_lineas = [
        f"{'─' * 50}",
        f"  RESUMEN EJECUTIVO - {model_name.upper()}",
        f"{'─' * 50}",
        "",
        f"  MÉTRICAS:",
        f"    Precision:     {precision_pct:6.1f}%",
        f"    Recall:        {recall_pct:6.1f}%",
        f"    F1 Score:      {f1_pct:6.1f}%",
        "",
        f"  DETECCIÓN (ventana {window_days}d):",
        f"    Detectadas:    {metrics['tp']:3d} / {total_fallas}",
        f"    No detectadas: {metrics['fn']:3d}",
        f"    Falsas alarmas:{metrics['fp']:3d}",
        "",
        f"  COSTOS:",
        f"    Sin modelo:   ${metrics['baseline']:6.1f}",
        f"    Con modelo:   ${metrics['cost']:6.1f}",
        f"    Ahorro:       ${ahorro:6.1f} ({metrics['savings_pct']:+.1f}%)",
        "",
        f"{'─' * 50}",
        f"  ESTADO: {estado}",
        f"{'─' * 50}",
    ]
    
    resumen_texto = '\n'.join(resumen_lineas)
    
    # Crear recuadro con el resumen
    props = dict(boxstyle='round,pad=0.8', facecolor='white', 
                 edgecolor=estado_color, linewidth=3)
    
    ax4.text(0.5, 0.5, resumen_texto, transform=ax4.transAxes, fontsize=11,
             verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace', bbox=props)
    
    # Título general
    plt.suptitle(f'{model_name.upper()} - MANTENIMIENTO PREDICTIVO\nResultados Finales (Ventana {window_days} días)', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.show()
    
    # Imprimir conclusión en texto
    print("\n" + "=" * 70)
    print(" CONCLUSIÓN")
    print("=" * 70)
    print(f"""
  El {model_name} detecta {metrics['tp']} de {total_fallas} fallas ({recall_pct:.1f}% recall)
  con una precisión del {precision_pct:.1f}%.
  
  IMPACTO ECONÓMICO:
    Sin modelo: ${metrics['baseline']:.1f} (todas las fallas cuestan)
    Con modelo: ${metrics['cost']:.1f}
    
  ══════════════════════════════════════════════════════════════════════
   AHORRO TOTAL: ${ahorro:.1f} ({metrics['savings_pct']:+.1f}%)
  ══════════════════════════════════════════════════════════════════════
""")
    print("=" * 70)
