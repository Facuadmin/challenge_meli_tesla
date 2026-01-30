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
