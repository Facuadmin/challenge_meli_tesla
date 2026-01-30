"""
Utilidades para EDA de Ofertas Relámpago
=========================================
Métodos para responder todas las preguntas del análisis y generar gráficos.

ESTRUCTURA DEL MÓDULO (6 TEMÁTICAS):
------------------------------------

TEMÁTICA 1: Performance y Resultados
    - PerformanceGeneral: Tasas de éxito, distribución sell-through
    - AnalisisZombies: Ofertas fallidas, GMV perdido

TEMÁTICA 2: Análisis Temporal
    - AnalisisTemporal: Horarios, duración, patrones diarios/semanales

TEMÁTICA 3: Categorías y Dominios
    - AnalisisCategoria: Verticales, dominios, Pareto, problemáticos
    - AnalisisCanibalizacion: Competencia interna

TEMÁTICA 4: Pricing, GMV y Velocidad
    - AnalisisPricing: Ticket promedio, precio vs performance
    - AnalisisVelocidad: GMV/hora, top/bottom performers

TEMÁTICA 5: Stock y Operaciones
    - AnalisisStock: Stock óptimo, eficiencia, sobreventas
    - AnalisisOrigen: Campo ORIGIN

TEMÁTICA 6: Estrategia e Impacto
    - AnalisisEnvio: Free shipping impact
    - AnalisisNegocio: Riesgo operativo, FOMO, productividad
    - AnalisisDerivado: COVID, dominios tóxicos, features predictivos

USO:
----
    from utils_eda import OfertasEDA, PerformanceGeneral, AnalisisTemporal, ...
    
    eda = OfertasEDA('path/to/data.csv')
    df = eda.df
    
    # Temática 1
    success_rates = PerformanceGeneral.get_success_rates(df)
    PerformanceGeneral.plot_success_rates(df)  # Muestra directamente

SCRIPT EJECUTABLE:
------------------
    Ver run_eda.py para ejecutar el análisis completo por temáticas.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Tuple, Dict, List, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN VISUAL GLOBAL
# =============================================================================

# Paleta de colores corporativa/consistente
COLORS = {
    'primary': '#2E86AB',      # Azul principal
    'secondary': '#A23B72',    # Magenta
    'success': '#28A745',      # Verde éxito
    'danger': '#DC3545',       # Rojo peligro
    'warning': '#FFC107',      # Amarillo advertencia
    'info': '#17A2B8',         # Cyan info
    'light': '#F8F9FA',        # Gris claro
    'dark': '#343A40',         # Gris oscuro
    'accent1': '#F18F01',      # Naranja
    'accent2': '#C73E1D',      # Rojo oscuro
    'accent3': '#3B1F2B',      # Púrpura oscuro
}

# Paleta secuencial para gráficos
COLOR_PALETTE = [
    COLORS['primary'], COLORS['secondary'], COLORS['success'], 
    COLORS['accent1'], COLORS['info'], COLORS['danger'],
    COLORS['warning'], COLORS['accent2']
]

# Configuración de estilo
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'axes.labelweight': 'medium',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.titleweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

# =============================================================================
# FUNCIONES DE FORMATEO
# =============================================================================

def format_metrics(metrics: Dict, title: str = None) -> str:
    """
    Formatea un diccionario de métricas de forma legible.
    
    - Números enteros: con separador de miles (1,234,567)
    - Tasas (0-1): como porcentaje (50.4%)
    - Decimales: redondeados a 2 decimales
    - Dinero (claves con 'gmv', 'amount', 'precio'): formato moneda
    
    Args:
        metrics: Diccionario con métricas
        title: Título opcional para el output
        
    Returns:
        String formateado listo para imprimir
    """
    lines = []
    if title:
        lines.append("=" * 50)
        lines.append(f"📊 {title.upper()}")
        lines.append("=" * 50)
    
    max_key_len = max(len(k) for k in metrics.keys()) if metrics else 0
    
    for key, value in metrics.items():
        key_display = key.replace('_', ' ').title()
        
        if isinstance(value, (int, np.integer)):
            formatted = f"{value:,}"
        elif isinstance(value, (float, np.floating)):
            # Detectar tasas (valores entre 0 y 1 con palabras clave)
            is_rate = any(word in key.lower() for word in ['tasa', 'rate', 'ratio', 'proporcion', 'pct'])
            is_money = any(word in key.lower() for word in ['gmv', 'amount', 'precio', 'ticket', 'revenue', 'monto'])
            
            if is_rate and 0 <= value <= 1:
                formatted = f"{value:.1%}"
            elif is_money:
                formatted = f"${value:,.2f}"
            elif abs(value) >= 1000:
                formatted = f"{value:,.0f}"
            else:
                formatted = f"{value:.4f}"
        else:
            formatted = str(value)
        
        lines.append(f"  {key_display:<{max_key_len + 5}} {formatted}")
    
    if title:
        lines.append("=" * 50)
    
    return "\n".join(lines)


def print_metrics(metrics: Dict, title: str = None) -> None:
    """Imprime un diccionario de métricas formateado."""
    print(format_metrics(metrics, title))


class MetricsDict(dict):
    """
    Diccionario que se muestra formateado al imprimirse.
    Hereda de dict para mantener compatibilidad total.
    """
    def __init__(self, *args, title: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._title = title
    
    def __repr__(self):
        if self._title:
            return format_metrics(self, self._title)
        # Sin título = ya se imprimió verbose, mostrar versión compacta
        return f"MetricsDict({len(self)} métricas) - usa print_metrics() para ver detalle"
    
    def __str__(self):
        if self._title:
            return format_metrics(self, self._title)
        # Para str() siempre mostrar formateado
        return format_metrics(self)
    
    def _format_verbose(self) -> str:
        """Formato verbose con emojis y estilo detallado."""
        lines = []
        lines.append("=" * 50)
        title = self._title or "Métricas"
        lines.append(f"📊 {title.upper()}")
        lines.append("=" * 50)
        
        # Separar totales de tasas para mejor legibilidad
        totals = {k: v for k, v in self.items() if not k.startswith('tasa_')}
        rates = {k: v for k, v in self.items() if k.startswith('tasa_')}
        
        # Emojis por tipo de métrica
        emoji_map = {
            'total': '📊', 'con_ventas': '✅', 'zombies': '❌',
            'sellout': '🎯', 'oversell': '⚠️', 'conversion': '✅',
            'zombie': '❌', 'gmv': '💰', 'revenue': '💰'
        }
        
        for key, value in totals.items():
            emoji = next((e for k, e in emoji_map.items() if k in key.lower()), '•')
            key_display = key.replace('_', ' ').title()
            if isinstance(value, (int, np.integer)):
                formatted = f"{value:,}"
            elif isinstance(value, float):
                formatted = f"{value:,.2f}"
            else:
                formatted = str(value)
            lines.append(f"{emoji} {key_display}: {formatted}")
        
        if rates:
            lines.append("-" * 50)
            for key, value in rates.items():
                emoji = next((e for k, e in emoji_map.items() if k in key.lower()), '📈')
                key_display = key.replace('tasa_', '').replace('_', ' ').title()
                lines.append(f"{emoji} Tasa {key_display}: {value:.1%}")
        
        lines.append("=" * 50)
        return "\n".join(lines)
sns.set_palette(COLOR_PALETTE)


def _add_value_labels(ax, fmt='.1f', suffix='', prefix='', fontsize=9, 
                      vertical=True, offset=0.02):
    """Helper para agregar etiquetas de valor a barras."""
    for container in ax.containers:
        if vertical:
            ax.bar_label(container, fmt=f'{prefix}%{{{fmt}}}{suffix}', 
                        fontsize=fontsize, padding=3)
        else:
            ax.bar_label(container, fmt=f'{prefix}%{{{fmt}}}{suffix}', 
                        fontsize=fontsize, padding=3)


# =============================================================================
# MAPEO DE CLASES A TEMÁTICAS
# =============================================================================

TEMATICAS_MAP = {
    1: {
        'nombre': 'Performance y Resultados',
        'clases': ['PerformanceGeneral', 'AnalisisZombies'],
        'preguntas': [
            '1.1 Tasas de éxito general',
            '1.2 Distribución del sell-through',
            '1.3 Análisis de ofertas fallidas'
        ]
    },
    2: {
        'nombre': 'Análisis Temporal',
        'clases': ['AnalisisTemporal'],
        'preguntas': [
            '2.1 Mejores horarios',
            '2.2 Duración óptima',
            '2.3 Patrones estacionales'
        ]
    },
    3: {
        'nombre': 'Categorías y Dominios',
        'clases': ['AnalisisCategoria', 'AnalisisCanibalizacion'],
        'preguntas': [
            '3.1 Performance por vertical',
            '3.2 Performance por dominio',
            '3.3 Relación sellout vs GMV',
            '3.4 Canibalización'
        ]
    },
    4: {
        'nombre': 'Pricing, GMV y Velocidad',
        'clases': ['AnalisisPricing', 'AnalisisVelocidad'],
        'preguntas': [
            '4.1 Ticket promedio por categoría',
            '4.2 Precio vs performance',
            '4.3 Métricas de velocidad'
        ]
    },
    5: {
        'nombre': 'Stock y Operaciones',
        'clases': ['AnalisisStock', 'AnalisisOrigen'],
        'preguntas': [
            '5.1 Stock óptimo',
            '5.2 Sobreventas',
            '5.3 Análisis de origen'
        ]
    },
    6: {
        'nombre': 'Estrategia e Impacto',
        'clases': ['AnalisisEnvio', 'AnalisisNegocio', 'AnalisisDerivado'],
        'preguntas': [
            '6.1 Impacto del envío gratis',
            '6.2 Riesgo operativo',
            '6.3 Unit economics',
            '6.4 Efecto FOMO',
            '6.5 Eficiencia del slot',
            '6.6 Productividad'
        ]
    }
}

# =============================================================================
# CLASE PRINCIPAL: OfertasEDA
# =============================================================================

class OfertasEDA:
    """Clase principal para el análisis exploratorio de ofertas relámpago."""
    
    def __init__(self, csv_path: str = None):
        """
        Inicializa el EDA cargando y preparando los datos.
        
        Args:
            csv_path: Ruta al archivo CSV. Si es None, busca en ubicaciones estándar.
        """
        self.df = None
        self.csv_path = csv_path
        if csv_path:
            self.load_data(csv_path)
    
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Carga el dataset desde un archivo CSV."""
        self.df = pd.read_csv(csv_path)
        self._prepare_data()
        return self.df
    
    def _prepare_data(self):
        """Prepara los datos: conversión de tipos y feature engineering."""
        df = self.df
        
        # Convertir fechas
        df['OFFER_START_DATE'] = pd.to_datetime(df['OFFER_START_DATE'])
        df['OFFER_START_DTTM'] = pd.to_datetime(df['OFFER_START_DTTM'])
        df['OFFER_FINISH_DTTM'] = pd.to_datetime(df['OFFER_FINISH_DTTM'])
        
        # Feature Engineering
        self._create_features()
    
    def _create_features(self):
        """Crea todas las features derivadas necesarias para el análisis."""
        df = self.df
        
        # Features temporales
        df['duration_hours'] = (df['OFFER_FINISH_DTTM'] - df['OFFER_START_DTTM']).dt.total_seconds() / 3600
        df['duration_minutes'] = df['duration_hours'] * 60
        df['start_hour'] = df['OFFER_START_DTTM'].dt.hour
        df['day_of_week'] = df['OFFER_START_DTTM'].dt.dayofweek
        df['day_name'] = df['OFFER_START_DTTM'].dt.day_name()
        df['week'] = df['OFFER_START_DTTM'].dt.isocalendar().week
        df['month'] = df['OFFER_START_DTTM'].dt.month
        df['is_weekend'] = df['day_of_week'] >= 5
        
        # Features de performance
        df['units_sold_by_stock'] = df['INVOLVED_STOCK'] - df['REMAINING_STOCK_AFTER_END']
        df['sell_through_rate'] = df['units_sold_by_stock'] / df['INVOLVED_STOCK']
        df['has_sales'] = df['SOLD_QUANTITY'].notna() & (df['SOLD_QUANTITY'] > 0)
        df['is_sold_out'] = df['REMAINING_STOCK_AFTER_END'] <= 0
        df['has_oversell'] = df['REMAINING_STOCK_AFTER_END'] < 0
        df['oversell_qty'] = df['REMAINING_STOCK_AFTER_END'].apply(lambda x: abs(min(0, x)))
        
        # Features de pricing
        df['avg_ticket'] = np.where(df['SOLD_QUANTITY'] > 0, 
                                     df['SOLD_AMOUNT'] / df['SOLD_QUANTITY'], np.nan)
        df['gmv_per_committed_unit'] = np.where(df['INVOLVED_STOCK'] > 0,
                                                 df['SOLD_AMOUNT'] / df['INVOLVED_STOCK'], np.nan)
        
        # Features de velocidad
        df['gmv_per_hour'] = np.where(df['duration_hours'] > 0, 
                                       df['SOLD_AMOUNT'] / df['duration_hours'], np.nan)
        df['units_per_hour'] = np.where(df['duration_hours'] > 0,
                                         df['units_sold_by_stock'] / df['duration_hours'], np.nan)
        
        # Features binarias
        df['has_free_shipping'] = df['SHIPPING_PAYMENT_TYPE'] == 'free_shipping'
        df['is_origin_A'] = df['ORIGIN'] == 'A'
        
        # Buckets de stock
        stock_bins = [0, 5, 10, 15, 20, 30, 50, 100, float('inf')]
        stock_labels = ['1-5', '6-10', '11-15', '16-20', '21-30', '31-50', '51-100', '>100']
        df['stock_bucket'] = pd.cut(df['INVOLVED_STOCK'], bins=stock_bins, labels=stock_labels)
        
        # Buckets de duración
        duration_bins = [0, 2, 4, 6, 8, 12, 24, float('inf')]
        duration_labels = ['0-2h', '2-4h', '4-6h', '6-8h', '8-12h', '12-24h', '>24h']
        df['duration_bucket'] = pd.cut(df['duration_hours'], bins=duration_bins, labels=duration_labels)
        
        # Buckets de sell-through
        str_bins = [0, 0.01, 0.25, 0.50, 0.75, 0.99, 1.0, float('inf')]
        str_labels = ['0% (zombie)', '1-25%', '25-50%', '50-75%', '75-99%', '100% (sellout)', '>100% (oversell)']
        df['str_bucket'] = pd.cut(df['sell_through_rate'], bins=str_bins, labels=str_labels, include_lowest=True)
        
        self.df = df


# =============================================================================
# TEMÁTICA 1: PERFORMANCE Y RESULTADOS
# =============================================================================
# Clases: PerformanceGeneral, AnalisisZombies
# Preguntas: Tasas de éxito, sell-through, ofertas fallidas
# =============================================================================

class PerformanceGeneral:
    """Métodos para analizar la performance general de las ofertas."""
    
    @staticmethod
    def get_success_rates(df: pd.DataFrame) -> Dict:
        """
        Calcula las tasas de éxito general de las ofertas.
        
        Args:
            df: DataFrame con los datos de ofertas
        
        Returns:
            MetricsDict con métricas formateadas automáticamente
            
        Note:
            Para ver gráfico + resumen completo usar plot_success_rates()
        """
        total = len(df)
        con_ventas = int(df['has_sales'].sum())
        zombies = total - con_ventas
        sellout = int(df['is_sold_out'].sum())
        oversell = int(df['has_oversell'].sum())
        
        tasa_conversion = con_ventas / total
        tasa_zombie = zombies / total
        tasa_sellout = sellout / total
        tasa_oversell = oversell / total
        
        return MetricsDict({
            'total_ofertas': total,
            'ofertas_con_ventas': con_ventas,
            'ofertas_zombies': zombies,
            'ofertas_sellout': sellout,
            'ofertas_oversell': oversell,
            'tasa_conversion': float(tasa_conversion),
            'tasa_zombie': float(tasa_zombie),
            'tasa_sellout': float(tasa_sellout),
            'tasa_oversell': float(tasa_oversell)
        }, title="Tasas de Éxito General")
    
    @staticmethod
    def plot_success_rates(df: pd.DataFrame, figsize: Tuple = (14, 5)) -> None:
        """
        Genera gráfico de donut chart y barras con las tasas de éxito.
        
        PREGUNTA: ¿Cuál es la tasa de éxito general de las ofertas relámpago?
        """
        metrics = PerformanceGeneral.get_success_rates(df)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('¿Cuál es la tasa de éxito general de las ofertas relámpago?', 
                    fontsize=14, fontweight='bold', color='#1A252F', y=1.02)
        
        # Gráfico 1: Donut Chart moderno
        labels = ['Con ventas', 'Zombies']
        sizes = [metrics['ofertas_con_ventas'], metrics['ofertas_zombies']]
        colors = [COLORS['success'], COLORS['danger']]
        
        # Crear donut chart
        wedges, texts, autotexts = axes[0].pie(
            sizes, labels=labels, autopct='%1.1f%%', colors=colors, 
            startangle=90, pctdistance=0.75,
            wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2),
            textprops={'fontsize': 11, 'fontweight': 'medium'}
        )
        for autotext in autotexts:
            autotext.set_fontsize(12)
            autotext.set_fontweight('bold')
        
        # Agregar texto central con total
        axes[0].text(0, 0, f'{metrics["total_ofertas"]:,}\nofertas', 
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    color=COLORS['dark'])
        axes[0].set_title('Distribución de Ofertas por Resultado', pad=15)
        
        # Gráfico 2: Zoom de ofertas CON ventas (100% = ofertas con ventas)
        # Categorías mutuamente excluyentes
        con_ventas = metrics['ofertas_con_ventas']
        sellout = metrics['ofertas_sellout']
        oversell = metrics['ofertas_oversell']
        
        # Desglose: ventas parciales, sellout exacto, oversell
        ventas_parciales = con_ventas - sellout  # vendieron algo pero no agotaron
        sellout_exacto = sellout - oversell       # agotaron pero no excedieron
        
        # Porcentajes sobre ofertas con ventas (nuevo 100%)
        pct_parciales = (ventas_parciales / con_ventas) * 100
        pct_sellout = (sellout_exacto / con_ventas) * 100
        pct_oversell = (oversell / con_ventas) * 100
        
        tasas = ['Ventas parciales\n(no agotaron)', 'Sellout\n(agotaron stock)', 'Oversell\n(exceso ventas)']
        valores = [pct_parciales, pct_sellout, pct_oversell]
        bar_colors = [COLORS['primary'], COLORS['success'], COLORS['danger']]
        
        bars = axes[1].barh(tasas, valores, color=bar_colors, height=0.6, 
                           edgecolor='white', linewidth=1)
        axes[1].set_xlabel('Porcentaje (%)', fontweight='medium')
        axes[1].set_title(f'Desglose de Ofertas con Ventas (n={con_ventas:,})', pad=15)
        axes[1].set_xlim(0, 105)
        
        # Agregar valores al final de las barras
        for bar, val in zip(bars, valores):
            axes[1].text(val + 1, bar.get_y() + bar.get_height()/2, 
                        f'{val:.1f}%', ha='left', va='center', 
                        fontsize=12, fontweight='bold', color=COLORS['dark'])
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        
        # Imprimir resumen
        print()
        print("=" * 60)
        print("📊 DISTRIBUCIÓN DE OFERTAS")
        print("=" * 60)
        print(f"Total ofertas: {metrics['total_ofertas']:,}")
        print("-" * 60)
        print(f"✅ Con ventas:    {metrics['ofertas_con_ventas']:,} ({metrics['tasa_conversion']:.1%})")
        print(f"❌ Zombies:       {metrics['ofertas_zombies']:,} ({metrics['tasa_zombie']:.1%})")
        print("-" * 60)
        print(f"📈 ZOOM: De las {con_ventas:,} ofertas con ventas:")
        print(f"   📦 Ventas parciales: {ventas_parciales:,} ({pct_parciales:.1f}%)")
        print(f"   🎯 Sellout exacto:   {sellout_exacto:,} ({pct_sellout:.1f}%)")
        print(f"   ⚠️  Oversell:        {oversell:,} ({pct_oversell:.1f}%)")
        print("=" * 60)
    
    @staticmethod
    def plot_sell_through_histogram(df: pd.DataFrame, figsize: Tuple = (12, 6)) -> None:
        """
        Genera histograma del sell-through rate con KDE overlay y zonas destacadas.
        
        PREGUNTA: ¿Cómo se distribuye el sell-through rate de las ofertas?
        """
        # Filtrar valores válidos
        str_valid = df['sell_through_rate'].dropna()
        str_valid = str_valid[str_valid <= 2]  # Cap en 200% para visualización
        
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle('¿Cómo se distribuye el sell-through rate de las ofertas?', 
                    fontsize=14, fontweight='bold', color='#1A252F', y=1.02)
        
        # Histograma con KDE
        n, bins, patches = ax.hist(str_valid, bins=50, edgecolor='white', 
                                   alpha=0.7, color=COLORS['primary'], density=True,
                                   label='Distribución')
        
        # Colorear barras por zona
        for i, (patch, left_edge) in enumerate(zip(patches, bins[:-1])):
            if left_edge < 0.01:
                patch.set_facecolor(COLORS['danger'])
                patch.set_alpha(0.8)
            elif left_edge >= 1.0:
                patch.set_facecolor(COLORS['success'])
                patch.set_alpha(0.8)
        
        # KDE overlay
        kde_x = np.linspace(str_valid.min(), str_valid.max(), 200)
        kde = stats.gaussian_kde(str_valid)
        ax.plot(kde_x, kde(kde_x), color=COLORS['accent1'], linewidth=2.5, 
                label='Densidad (KDE)')
        
        # Líneas de referencia con zonas sombreadas
        ax.axvline(x=0, color=COLORS['danger'], linestyle='--', linewidth=2, alpha=0.8)
        ax.axvline(x=1, color=COLORS['success'], linestyle='--', linewidth=2, alpha=0.8)
        
        # Sombrear zona de zombies
        ax.axvspan(-0.05, 0.05, alpha=0.15, color=COLORS['danger'], label='Zona Zombies')
        # Sombrear zona de sellout
        ax.axvspan(0.95, 1.05, alpha=0.15, color=COLORS['success'], label='Zona Sellout')
        
        # Agregar anotaciones de percentiles
        median = str_valid.median()
        ax.axvline(x=median, color=COLORS['secondary'], linestyle='-', linewidth=2, alpha=0.8)
        ax.annotate(f'Mediana\n{median:.0%}', xy=(median, ax.get_ylim()[1]*0.85),
                   fontsize=10, ha='center', fontweight='bold', color=COLORS['secondary'],
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Sell-Through Rate', fontweight='medium')
        ax.set_ylabel('Densidad', fontweight='medium')
        ax.set_title('Distribución del Sell-Through Rate', pad=15)
        ax.legend(loc='upper right', framealpha=0.9)
        ax.set_xlim(-0.1, 1.5)
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)


# =============================================================================
# TEMÁTICA 2: ANÁLISIS TEMPORAL
# =============================================================================
# Clases: AnalisisTemporal
# Preguntas: Horarios, duración, patrones diarios/semanales
# =============================================================================

class AnalisisTemporal:
    """Métodos para análisis temporal de las ofertas."""
    
    @staticmethod
    def get_hourly_performance(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula la performance por hora de inicio.
        
        Returns:
            DataFrame con métricas por hora
        """
        hourly = df.groupby('start_hour').agg({
            'OFFER_START_DATE': 'count',
            'has_sales': ['sum', 'mean'],
            'is_sold_out': 'mean',
            'SOLD_AMOUNT': 'sum',
            'sell_through_rate': 'mean'
        })
        hourly.columns = ['total_ofertas', 'con_ventas', 'tasa_conversion', 
                          'tasa_sellout', 'gmv_total', 'sell_through_rate']
        hourly['gmv_promedio'] = hourly['gmv_total'] / hourly['con_ventas']
        return hourly.round(4)
    
    @staticmethod
    def get_daily_performance(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula la performance por día de la semana.
        """
        daily = df.groupby('day_name').agg({
            'OFFER_START_DATE': 'count',
            'has_sales': ['sum', 'mean'],
            'is_sold_out': 'mean',
            'SOLD_AMOUNT': 'sum',
            'sell_through_rate': 'mean'
        })
        daily.columns = ['total_ofertas', 'con_ventas', 'tasa_conversion',
                         'tasa_sellout', 'gmv_total', 'sell_through_rate']
        
        # Ordenar por día de semana
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily = daily.reindex(day_order)
        return daily.round(4)
    
    @staticmethod
    def get_duration_performance(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula la performance por rangos de duración.
        """
        duration_perf = df.groupby('duration_bucket').agg({
            'OFFER_START_DATE': 'count',
            'has_sales': 'mean',
            'is_sold_out': 'mean',
            'sell_through_rate': 'mean',
            'gmv_per_hour': 'mean'
        })
        duration_perf.columns = ['total_ofertas', 'tasa_conversion', 'tasa_sellout',
                                  'sell_through_rate', 'gmv_por_hora']
        return duration_perf.round(4)
    
    @staticmethod
    def get_date_trends(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula tendencias diarias a lo largo del período.
        """
        daily_stats = df.groupby('OFFER_START_DATE').agg({
            'OFFER_START_DTTM': 'count',
            'has_sales': 'mean',
            'SOLD_AMOUNT': 'sum',
            'is_sold_out': 'mean'
        }).rename(columns={
            'OFFER_START_DTTM': 'num_ofertas',
            'has_sales': 'tasa_conv',
            'SOLD_AMOUNT': 'gmv',
            'is_sold_out': 'tasa_sellout'
        })
        return daily_stats
    
    @staticmethod
    def plot_hourly_heatmap(df: pd.DataFrame, figsize: Tuple = (16, 8)) -> None:
        """
        Genera heatmap de performance por día de semana y hora con resaltado de extremos.
        
        PREGUNTA: ¿Cuál es la mejor combinación de día y hora para lanzar ofertas?
        """
        # Crear pivot table
        pivot = df.pivot_table(
            values='has_sales',
            index='day_name',
            columns='start_hour',
            aggfunc='mean'
        )
        
        # Ordenar días con nombres en español
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_names_es = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
        pivot = pivot.reindex(day_order)
        
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle('¿Cuál es la mejor combinación de día y hora para lanzar ofertas?', 
                    fontsize=14, fontweight='bold', color='#1A252F', y=1.02)
        
        # Crear heatmap con mejor paleta
        hm = sns.heatmap(pivot, annot=True, fmt='.0%', cmap='RdYlGn', ax=ax,
                    cbar_kws={'label': 'Tasa de Conversión', 'shrink': 0.8},
                    linewidths=0.5, linecolor='white',
                    annot_kws={'fontsize': 9, 'fontweight': 'medium'})
        
        # Identificar y resaltar máximo y mínimo
        max_val = pivot.max().max()
        min_val = pivot.min().min()
        
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.iloc[i, j]
                if pd.notna(val):
                    if val == max_val:
                        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, 
                                    edgecolor=COLORS['success'], linewidth=3))
                        ax.annotate('★', (j+0.5, i+0.15), fontsize=12, 
                                   ha='center', va='center', color='darkgreen', fontweight='bold')
                    elif val == min_val:
                        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                    edgecolor=COLORS['danger'], linewidth=3))
                        ax.annotate('✗', (j+0.5, i+0.15), fontsize=12,
                                   ha='center', va='center', color='darkred', fontweight='bold')
        
        # Agregar fila de promedios
        avg_by_hour = pivot.mean().values
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks([i + 0.5 for i in range(len(pivot.columns))])
        ax2.set_xticklabels([f'{v:.0%}' for v in avg_by_hour], fontsize=8, color=COLORS['secondary'])
        ax2.set_xlabel('Promedio por Hora', fontsize=10, color=COLORS['secondary'], fontweight='medium')
        
        ax.set_title('Tasa de Conversión por Día y Hora\n★ Mejor combinación  ✗ Peor combinación', 
                    pad=25, fontsize=14)
        ax.set_xlabel('Hora de Inicio', fontweight='medium')
        ax.set_ylabel('Día de la Semana', fontweight='medium')
        ax.set_yticklabels(day_names_es, rotation=0)
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        
        # Imprimir resumen
        best_hour = pivot.mean().idxmax()
        best_day = pivot.mean(axis=1).idxmax()
        worst_hour = pivot.mean().idxmin()
        worst_day = pivot.mean(axis=1).idxmin()
        best_combo = pivot.stack().idxmax()
        worst_combo = pivot.stack().idxmin()
        print()
        print("=" * 60)
        print("🗓️ HEATMAP: CONVERSIÓN POR DÍA Y HORA")
        print("=" * 60)
        print(f"🏆 Mejor hora (promedio):    {best_hour}:00 ({pivot.mean()[best_hour]:.1%})")
        print(f"🏆 Mejor día (promedio):     {best_day} ({pivot.mean(axis=1)[best_day]:.1%})")
        print(f"📉 Peor hora (promedio):     {worst_hour}:00 ({pivot.mean()[worst_hour]:.1%})")
        print(f"📉 Peor día (promedio):      {worst_day} ({pivot.mean(axis=1)[worst_day]:.1%})")
        print("-" * 60)
        print(f"🌟 Mejor combo:  {best_combo[0]} {best_combo[1]}:00 ({pivot.loc[best_combo[0], best_combo[1]]:.1%})")
        print(f"💀 Peor combo:   {worst_combo[0]} {worst_combo[1]}:00 ({pivot.loc[worst_combo[0], worst_combo[1]]:.1%})")
        print("=" * 60)
    
    @staticmethod
    def plot_hourly_performance(df: pd.DataFrame, figsize: Tuple = (12, 5)) -> None:
        """
        Genera gráfico de barras de performance por hora.
        
        PREGUNTA: ¿Cuáles son las mejores horas para lanzar ofertas?
        """
        hourly = AnalisisTemporal.get_hourly_performance(df)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('¿Cuáles son las mejores horas para lanzar ofertas?', 
                    fontsize=14, fontweight='bold', color='#1A252F', y=1.02)
        
        # Gráfico 1: Volumen de ofertas por hora
        axes[0].bar(hourly.index, hourly['total_ofertas'], color='#3498db', alpha=0.7)
        axes[0].set_xlabel('Hora de Inicio')
        axes[0].set_ylabel('Número de Ofertas')
        axes[0].set_title('Volumen de Ofertas por Hora')
        
        # Gráfico 2: Tasa de conversión por hora
        axes[1].bar(hourly.index, hourly['tasa_conversion'] * 100, color='#2ecc71', alpha=0.7)
        axes[1].axhline(y=hourly['tasa_conversion'].mean() * 100, color='red', 
                        linestyle='--', label=f'Promedio: {hourly["tasa_conversion"].mean()*100:.1f}%')
        axes[1].set_xlabel('Hora de Inicio')
        axes[1].set_ylabel('Tasa de Conversión (%)')
        axes[1].set_title('Tasa de Conversión por Hora')
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        
        # Imprimir resumen
        best_hour = hourly['tasa_conversion'].idxmax()
        worst_hour = hourly['tasa_conversion'].idxmin()
        busiest_hour = hourly['total_ofertas'].idxmax()
        print()
        print("=" * 60)
        print("⏰ PERFORMANCE POR HORA DE INICIO")
        print("=" * 60)
        print(f"Total ofertas analizadas: {hourly['total_ofertas'].sum():,}")
        print("-" * 60)
        print(f"🏆 Mejor hora (conversión):    {best_hour}:00 ({hourly.loc[best_hour, 'tasa_conversion']:.1%})")
        print(f"📉 Peor hora (conversión):     {worst_hour}:00 ({hourly.loc[worst_hour, 'tasa_conversion']:.1%})")
        print(f"📊 Hora más activa:            {busiest_hour}:00 ({hourly.loc[busiest_hour, 'total_ofertas']:,} ofertas)")
        print(f"📊 Promedio conversión:        {hourly['tasa_conversion'].mean():.1%}")
        print("=" * 60)
    
    @staticmethod
    def plot_duration_analysis(df: pd.DataFrame, figsize: Tuple = (14, 7)) -> None:
        """
        Genera gráfico de análisis de duración con diseño moderno y profesional.
        
        PREGUNTA: ¿Cuál es la duración óptima para una oferta?
        """
        duration_perf = AnalisisTemporal.get_duration_performance(df)
        
        # Configuración de estilo
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        ax.set_facecolor('#FAFAFA')
        
        x = np.arange(len(duration_perf))
        width = 0.38
        
        # Paleta de colores profesional con gradiente
        color_conversion = '#4A90D9'  # Azul profesional
        color_sellout = '#50C878'      # Verde esmeralda
        
        # Crear barras con efecto sutil
        bars1 = ax.bar(x - width/2, duration_perf['tasa_conversion'] * 100, 
                       width, label='Tasa Conversión', color=color_conversion,
                       edgecolor='white', linewidth=1.5, alpha=0.9, zorder=3)
        bars2 = ax.bar(x + width/2, duration_perf['tasa_sellout'] * 100,
                       width, label='Tasa Sellout', color=color_sellout,
                       edgecolor='white', linewidth=1.5, alpha=0.9, zorder=3)
        
        # Agregar valores sobre las barras
        def add_bar_labels(bars, color, offset=0):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 5),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=9, fontweight='bold',
                           color=color, alpha=0.85)
        
        add_bar_labels(bars1, color_conversion)
        add_bar_labels(bars2, color_sellout)
        
        # Líneas de promedio con estilo mejorado
        avg_conversion = duration_perf['tasa_conversion'].mean() * 100
        avg_sellout = duration_perf['tasa_sellout'].mean() * 100
        
        ax.axhline(y=avg_conversion, color=color_conversion, linestyle='--', 
                   alpha=0.5, linewidth=1.5, zorder=2)
        ax.axhline(y=avg_sellout, color=color_sellout, linestyle='--', 
                   alpha=0.5, linewidth=1.5, zorder=2)
        
        # Anotaciones de promedios en el lado derecho
        ax.text(len(duration_perf) - 0.3, avg_conversion + 0.8, f'Prom: {avg_conversion:.1f}%', 
                fontsize=8, color=color_conversion, alpha=0.8, fontweight='medium')
        ax.text(len(duration_perf) - 0.3, avg_sellout + 0.8, f'Prom: {avg_sellout:.1f}%', 
                fontsize=8, color=color_sellout, alpha=0.8, fontweight='medium')
        
        # Configuración de ejes con estilo mejorado
        ax.set_xlabel('Duración de la Oferta', fontsize=12, fontweight='medium', 
                      labelpad=10, color='#2C3E50')
        ax.set_ylabel('Porcentaje (%)', fontsize=12, fontweight='medium', 
                      labelpad=10, color='#2C3E50')
        ax.set_title('¿Cuál es la duración óptima para una oferta?', fontsize=16, 
                     fontweight='bold', pad=20, color='#1A252F')
        
        ax.set_xticks(x)
        ax.set_xticklabels(duration_perf.index, rotation=30, ha='right', 
                          fontsize=10, color='#34495E')
        ax.tick_params(axis='y', labelsize=10, colors='#34495E')
        
        # Ajustar límites para dar espacio a las etiquetas
        max_val = max(duration_perf['tasa_conversion'].max(), duration_perf['tasa_sellout'].max()) * 100
        ax.set_ylim(0, max_val * 1.18)
        
        # Grid más sutil
        ax.yaxis.grid(True, linestyle='-', alpha=0.3, color='#BDC3C7', zorder=1)
        ax.xaxis.grid(False)
        
        # Leyenda con estilo mejorado
        legend = ax.legend(loc='upper right', frameon=True, fancybox=True,
                          shadow=False, fontsize=10, framealpha=0.95,
                          edgecolor='#E0E0E0')
        legend.get_frame().set_linewidth(1)
        
        # Remover bordes superior y derecho para look más limpio
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#BDC3C7')
        ax.spines['bottom'].set_color('#BDC3C7')
        
        # Agregar indicadores de mejor performance
        best_conv_idx = duration_perf['tasa_conversion'].idxmax()
        best_sellout_idx = duration_perf['tasa_sellout'].idxmax()
        
        # Posiciones de los mejores
        best_conv_pos = list(duration_perf.index).index(best_conv_idx)
        best_sellout_pos = list(duration_perf.index).index(best_sellout_idx)
        
        # Marcadores de estrella para los mejores
        ax.scatter(best_conv_pos - width/2, duration_perf.loc[best_conv_idx, 'tasa_conversion'] * 100 + max_val * 0.08,
                  marker='*', s=150, color='#F39C12', zorder=5, edgecolors='white', linewidths=0.5)
        ax.scatter(best_sellout_pos + width/2, duration_perf.loc[best_sellout_idx, 'tasa_sellout'] * 100 + max_val * 0.08,
                  marker='*', s=150, color='#F39C12', zorder=5, edgecolors='white', linewidths=0.5)
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        
        # Imprimir resumen
        best_duration = duration_perf['tasa_conversion'].idxmax()
        best_sellout = duration_perf['tasa_sellout'].idxmax()
        most_common = duration_perf['total_ofertas'].idxmax()
        print()
        print("=" * 60)
        print("⏱️ PERFORMANCE POR DURACIÓN DE OFERTA")
        print("=" * 60)
        print(f"Total ofertas: {int(duration_perf['total_ofertas'].sum()):,}")
        print("-" * 60)
        print(f"🏆 Mejor duración (conversión): {best_duration} ({duration_perf.loc[best_duration, 'tasa_conversion']:.1%})")
        print(f"🎯 Mejor duración (sellout):    {best_sellout} ({duration_perf.loc[best_sellout, 'tasa_sellout']:.1%})")
        print(f"📊 Duración más común:          {most_common} ({int(duration_perf.loc[most_common, 'total_ofertas']):,} ofertas)")
        print("-" * 60)
        print("📋 Resumen por duración:")
        for idx in duration_perf.index:
            print(f"   {idx}: Conv {duration_perf.loc[idx, 'tasa_conversion']:.1%} | Sellout {duration_perf.loc[idx, 'tasa_sellout']:.1%}")
        print("=" * 60)
    
    @staticmethod
    def plot_dual_axis_hourly(df: pd.DataFrame, figsize: Tuple = (14, 6)) -> None:
        """
        Genera gráfico de doble eje: volumen (barras) y conversión (línea) por hora.
        
        PREGUNTA: ¿Cómo se relaciona el volumen de ofertas con la conversión por hora?
        """
        # Agregar datos por hora
        hourly = df.groupby('start_hour').agg(
            total_ofertas=('OFFER_START_DATE', 'count'),
            tasa_conversion=('has_sales', 'mean')
        ).reset_index()
        
        fig, ax1 = plt.subplots(figsize=figsize)
        
        # Barras para volumen
        bars = ax1.bar(hourly['start_hour'], hourly['total_ofertas'], 
                       color='steelblue', alpha=0.7, label='Volumen')
        ax1.set_xlabel('Hora de Inicio')
        ax1.set_ylabel('Número de Ofertas', color='steelblue')
        ax1.tick_params(axis='y', labelcolor='steelblue')
        
        # Línea para conversión (eje secundario)
        ax2 = ax1.twinx()
        line = ax2.plot(hourly['start_hour'], hourly['tasa_conversion'] * 100, 
                        color='red', linewidth=2.5, marker='o', label='Conversión')
        ax2.set_ylabel('Tasa de Conversión (%)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        avg = hourly['tasa_conversion'].mean() * 100
        ax2.axhline(y=avg, color='red', linestyle='--', alpha=0.5, label=f'Promedio: {avg:.1f}%')
        
        plt.title('¿Cómo se relaciona el volumen de ofertas con la conversión por hora?', 
                 fontsize=14, fontweight='bold', color='#1A252F')
        fig.tight_layout()
        plt.show()
        plt.close(fig)
        
        # Imprimir resumen
        best_hour_conv = hourly.loc[hourly['tasa_conversion'].idxmax(), 'start_hour']
        worst_hour_conv = hourly.loc[hourly['tasa_conversion'].idxmin(), 'start_hour']
        busiest_hour = hourly.loc[hourly['total_ofertas'].idxmax(), 'start_hour']
        quietest_hour = hourly.loc[hourly['total_ofertas'].idxmin(), 'start_hour']
        avg_conv = hourly['tasa_conversion'].mean()
        above_avg = hourly[hourly['tasa_conversion'] > avg_conv]['start_hour'].tolist()
        print()
        print("=" * 60)
        print("⏰ VOLUMEN vs CONVERSIÓN POR HORA (DUAL AXIS)")
        print("=" * 60)
        print(f"Total ofertas: {hourly['total_ofertas'].sum():,}")
        print(f"Promedio conversión: {avg_conv:.1%}")
        print("-" * 60)
        print(f"🏆 Mejor hora (conversión):  {int(best_hour_conv)}:00 ({hourly.loc[hourly['start_hour'] == best_hour_conv, 'tasa_conversion'].values[0]:.1%})")
        print(f"📉 Peor hora (conversión):   {int(worst_hour_conv)}:00 ({hourly.loc[hourly['start_hour'] == worst_hour_conv, 'tasa_conversion'].values[0]:.1%})")
        print(f"📊 Hora más activa:          {int(busiest_hour)}:00 ({hourly.loc[hourly['start_hour'] == busiest_hour, 'total_ofertas'].values[0]:,} ofertas)")
        print(f"📊 Hora menos activa:        {int(quietest_hour)}:00 ({hourly.loc[hourly['start_hour'] == quietest_hour, 'total_ofertas'].values[0]:,} ofertas)")
        print("-" * 60)
        print(f"✅ Horas sobre promedio: {', '.join([f'{int(h)}:00' for h in above_avg])}")
        print("=" * 60)
    
    @staticmethod
    def plot_hourly_opportunity_score(df: pd.DataFrame, 
                                       weight_conversion: float = 0.6,
                                       figsize: Tuple = (16, 6)) -> None:
        """
        Calcula y visualiza un Score de Oportunidad que balancea conversión y volumen.
        
        La métrica combina:
        - Tasa de conversión normalizada (calidad)
        - Volumen de ofertas normalizado (confianza/representatividad)
        
        PREGUNTA: ¿Qué horas ofrecen el mejor balance entre volumen y conversión?
        
        Args:
            df: DataFrame con los datos
            weight_conversion: Peso para conversión (0-1). Default 0.6 = 60% conversión, 40% volumen
            figsize: Tamaño del gráfico
        """
        weight_volume = 1 - weight_conversion
        
        # Agregar datos por hora
        hourly = df.groupby('start_hour').agg(
            total_ofertas=('OFFER_START_DATE', 'count'),
            ofertas_con_ventas=('has_sales', 'sum'),
            tasa_conversion=('has_sales', 'mean'),
            gmv_total=('SOLD_AMOUNT', 'sum')
        ).reset_index()
        
        # Normalizar métricas (0-1) usando min-max scaling
        hourly['conversion_norm'] = (hourly['tasa_conversion'] - hourly['tasa_conversion'].min()) / \
                                    (hourly['tasa_conversion'].max() - hourly['tasa_conversion'].min())
        hourly['volumen_norm'] = (hourly['total_ofertas'] - hourly['total_ofertas'].min()) / \
                                  (hourly['total_ofertas'].max() - hourly['total_ofertas'].min())
        
        # Calcular Score de Oportunidad (media ponderada)
        hourly['opportunity_score'] = (weight_conversion * hourly['conversion_norm'] + 
                                        weight_volume * hourly['volumen_norm'])
        
        # Ordenar por score
        hourly_ranked = hourly.sort_values('opportunity_score', ascending=False).reset_index(drop=True)
        hourly_ranked['ranking'] = range(1, len(hourly_ranked) + 1)
        
        # --- Visualización ---
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('¿Qué horas ofrecen el mejor balance entre volumen y conversión?', 
                    fontsize=14, fontweight='bold', color='#1A252F', y=1.02)
        
        # 1. Scatter: Conversión vs Volumen con tamaño = score
        ax1 = axes[0]
        scatter = ax1.scatter(hourly['total_ofertas'], 
                              hourly['tasa_conversion'] * 100,
                              s=hourly['opportunity_score'] * 500 + 50,
                              c=hourly['opportunity_score'],
                              cmap='RdYlGn',
                              alpha=0.8,
                              edgecolors='white',
                              linewidth=1.5)
        
        # Anotar cada punto con la hora
        for _, row in hourly.iterrows():
            ax1.annotate(f"{int(row['start_hour'])}h", 
                        (row['total_ofertas'], row['tasa_conversion'] * 100),
                        fontsize=9, ha='center', va='bottom', fontweight='bold')
        
        ax1.set_xlabel('Número de Ofertas (Volumen)')
        ax1.set_ylabel('Tasa de Conversión (%)')
        ax1.set_title('Trade-off: Conversión vs Volumen\n(tamaño/color = Score de Oportunidad)')
        plt.colorbar(scatter, ax=ax1, label='Score')
        
        # Marcar zona óptima (cuadrante superior derecho)
        median_vol = hourly['total_ofertas'].median()
        median_conv = hourly['tasa_conversion'].median() * 100
        ax1.axvline(x=median_vol, color='gray', linestyle='--', alpha=0.5)
        ax1.axhline(y=median_conv, color='gray', linestyle='--', alpha=0.5)
        ax1.text(ax1.get_xlim()[1] * 0.95, ax1.get_ylim()[1] * 0.95, 'ÓPTIMO', 
                ha='right', va='top', fontsize=10, color='green', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        # 2. Barras horizontales: Ranking por Score
        ax2 = axes[1]
        colors = plt.cm.RdYlGn(hourly_ranked['opportunity_score'])
        bars = ax2.barh([f"{int(h)}:00" for h in hourly_ranked['start_hour']], 
                        hourly_ranked['opportunity_score'],
                        color=colors, edgecolor='white')
        ax2.set_xlabel('Score de Oportunidad')
        ax2.set_title(f'Ranking de Horas\n(Peso: {weight_conversion:.0%} conversión, {weight_volume:.0%} volumen)')
        ax2.set_xlim(0, 1.1)
        
        # Agregar valores
        for bar, score in zip(bars, hourly_ranked['opportunity_score']):
            ax2.text(score + 0.02, bar.get_y() + bar.get_height()/2, 
                    f'{score:.2f}', va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        
        # Imprimir resumen
        best_hour = hourly_ranked.iloc[0]
        worst_hour = hourly_ranked.iloc[-1]
        
        print()
        print("=" * 70)
        print("🎯 SCORE DE OPORTUNIDAD POR HORA")
        print(f"   Fórmula: {weight_conversion:.0%} × Conversión_norm + {weight_volume:.0%} × Volumen_norm")
        print("=" * 70)
        print()
        print("🏆 MEJORES HORAS (balance conversión + volumen):")
        for i, (_, row) in enumerate(hourly_ranked.head(3).iterrows(), 1):
            print(f"   {i}. {int(row['start_hour'])}:00 → Score: {row['opportunity_score']:.3f} "
                  f"(Conv: {row['tasa_conversion']:.1%}, Ofertas: {int(row['total_ofertas']):,})")
        print()
        print("⚠️  HORAS A EVITAR:")
        for _, row in hourly_ranked.tail(3).iterrows():
            print(f"   • {int(row['start_hour'])}:00 → Score: {row['opportunity_score']:.3f} "
                  f"(Conv: {row['tasa_conversion']:.1%}, Ofertas: {int(row['total_ofertas']):,})")
        print()
        print("-" * 70)
        print("💡 INTERPRETACIÓN:")
        print("   • Score alto = buena conversión Y volumen representativo")
        print("   • Score bajo = mala conversión O pocas ofertas (no confiable)")
        print("   • Ajusta weight_conversion para priorizar calidad vs cantidad")
        print("=" * 70)
    
    @staticmethod
    def plot_daily_trends(df: pd.DataFrame, figsize: Tuple = (14, 6)) -> None:
        """
        Genera gráfico de tendencias diarias.
        
        PREGUNTA: ¿Cómo han evolucionado las ofertas a lo largo del tiempo?
        """
        daily = AnalisisTemporal.get_date_trends(df)
        
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        fig.suptitle('¿Cómo han evolucionado las ofertas a lo largo del tiempo?', 
                    fontsize=14, fontweight='bold', color='#1A252F', y=1.02)
        
        # Gráfico 1: Volumen de ofertas
        axes[0].plot(daily.index, daily['num_ofertas'], color='#3498db', marker='o', markersize=3)
        axes[0].fill_between(daily.index, daily['num_ofertas'], alpha=0.3)
        axes[0].set_ylabel('Número de Ofertas')
        axes[0].set_title('Tendencias Diarias')
        
        # Gráfico 2: Tasa de conversión
        axes[1].plot(daily.index, daily['tasa_conv'] * 100, color='#2ecc71', marker='o', markersize=3)
        axes[1].axhline(y=daily['tasa_conv'].mean() * 100, color='red', linestyle='--', alpha=0.7)
        axes[1].set_ylabel('Tasa de Conversión (%)')
        axes[1].set_xlabel('Fecha')
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        
        # Imprimir resumen
        best_day = daily['tasa_conv'].idxmax()
        worst_day = daily['tasa_conv'].idxmin()
        busiest_day = daily['num_ofertas'].idxmax()
        print()
        print("=" * 60)
        print("📅 TENDENCIAS DIARIAS")
        print("=" * 60)
        print(f"Período: {daily.index.min().strftime('%Y-%m-%d')} a {daily.index.max().strftime('%Y-%m-%d')}")
        print(f"Total días: {len(daily)}")
        print("-" * 60)
        print(f"📈 Mejor día (conversión):  {best_day.strftime('%Y-%m-%d')} ({daily.loc[best_day, 'tasa_conv']:.1%})")
        print(f"📉 Peor día (conversión):   {worst_day.strftime('%Y-%m-%d')} ({daily.loc[worst_day, 'tasa_conv']:.1%})")
        print(f"📊 Día más activo:          {busiest_day.strftime('%Y-%m-%d')} ({int(daily.loc[busiest_day, 'num_ofertas']):,} ofertas)")
        print("-" * 60)
        print(f"Promedio ofertas/día: {daily['num_ofertas'].mean():.0f}")
        print(f"Promedio conversión:  {daily['tasa_conv'].mean():.1%}")
        print(f"GMV total período:    ${daily['gmv'].sum():,.0f}")
        print("=" * 60)


# =============================================================================
# TEMÁTICA 3: CATEGORÍAS Y DOMINIOS (Parte 1)
# =============================================================================
# Clases: AnalisisCategoria, AnalisisCanibalizacion
# Preguntas: Verticales, dominios, Pareto, canibalización
# =============================================================================

class AnalisisCategoria:
    """Métodos para análisis por categoría (Vertical, Dominio)."""
    
    @staticmethod
    def get_vertical_performance(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula la performance por vertical.
        """
        vertical = df.groupby('VERTICAL').agg({
            'OFFER_START_DATE': 'count',
            'has_sales': ['sum', 'mean'],
            'is_sold_out': 'mean',
            'SOLD_AMOUNT': 'sum',
            'sell_through_rate': 'mean',
            'avg_ticket': 'mean'
        })
        vertical.columns = ['total_ofertas', 'con_ventas', 'tasa_conversion',
                            'tasa_sellout', 'gmv_total', 'sell_through_rate', 'ticket_promedio']
        vertical['gmv_por_oferta'] = vertical['gmv_total'] / vertical['con_ventas']
        return vertical.sort_values('gmv_total', ascending=False).round(4)
    
    @staticmethod
    def get_domain_agg_performance(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula la performance por DOM_DOMAIN_AGG1.
        """
        domain_agg = df.groupby('DOM_DOMAIN_AGG1').agg({
            'OFFER_START_DATE': 'count',
            'has_sales': ['sum', 'mean'],
            'is_sold_out': 'mean',
            'SOLD_AMOUNT': 'sum',
            'sell_through_rate': 'mean',
            'avg_ticket': 'mean'
        })
        domain_agg.columns = ['total_ofertas', 'con_ventas', 'tasa_conversion',
                              'tasa_sellout', 'gmv_total', 'sell_through_rate', 'ticket_promedio']
        domain_agg['gmv_por_oferta'] = domain_agg['gmv_total'] / domain_agg['con_ventas']
        return domain_agg.sort_values('gmv_total', ascending=False).round(4)
    
    @staticmethod
    def get_domain_performance(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        """
        Calcula la performance por DOM_DOMAIN_AGG1 (top N).
        """
        domain = df.groupby('DOM_DOMAIN_AGG1').agg({
            'OFFER_START_DATE': 'count',
            'has_sales': 'mean',
            'is_sold_out': 'mean',
            'SOLD_AMOUNT': 'sum',
            'SOLD_QUANTITY': 'sum',
            'sell_through_rate': 'mean'
        })
        domain.columns = ['total_ofertas', 'tasa_conversion', 'tasa_sellout',
                          'gmv_total', 'unidades_vendidas', 'sell_through_rate']
        return domain.sort_values('gmv_total', ascending=False).head(top_n).round(4)
    
    @staticmethod
    def get_pareto_analysis(df: pd.DataFrame) -> Dict:
        """
        Análisis de concentración Pareto del GMV por dominio.
        
        Returns:
            Dict con métricas de concentración
        """
        domain_gmv = df.groupby('DOM_DOMAIN_AGG1')['SOLD_AMOUNT'].sum().sort_values(ascending=False)
        cumsum = domain_gmv.cumsum() / domain_gmv.sum()
        
        # Cuántos dominios para 80% del GMV
        n_80 = (cumsum < 0.8).sum() + 1
        n_90 = (cumsum < 0.9).sum() + 1
        
        return {
            'total_dominios': len(domain_gmv),
            'dominios_80_gmv': n_80,
            'pct_dominios_80_gmv': n_80 / len(domain_gmv) * 100,
            'dominios_90_gmv': n_90,
            'pct_dominios_90_gmv': n_90 / len(domain_gmv) * 100,
            'top10_gmv_pct': domain_gmv.head(10).sum() / domain_gmv.sum() * 100,
            'gmv_cumsum': cumsum
        }
    
    @staticmethod
    def get_problematic_domains(df: pd.DataFrame, min_offers: int = 20, 
                                 max_success_rate: float = 0.2) -> pd.DataFrame:
        """
        Identifica dominios problemáticos (bajo éxito).
        
        Args:
            min_offers: Mínimo de ofertas para considerar
            max_success_rate: Máxima tasa de éxito para ser problemático
        """
        domain_success = df.groupby('DOM_DOMAIN_AGG1').agg({
            'has_sales': ['sum', 'count', 'mean'],
            'INVOLVED_STOCK': 'sum'
        })
        domain_success.columns = ['ventas_ok', 'total_ofertas', 'tasa_exito', 'stock_comprometido']
        domain_success = domain_success[domain_success['total_ofertas'] >= min_offers]
        problematic = domain_success[domain_success['tasa_exito'] < max_success_rate]
        return problematic.sort_values('total_ofertas', ascending=False).round(4)
    
    @staticmethod
    def get_sellout_vs_gmv_by_category(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula la relación entre tasa de agotamiento y GMV por categoría.
        """
        cat_perf = df.groupby('DOM_DOMAIN_AGG1').agg({
            'is_sold_out': 'mean',
            'SOLD_AMOUNT': ['sum', 'mean'],
            'has_sales': 'mean'
        })
        cat_perf.columns = ['tasa_sellout', 'gmv_total', 'gmv_promedio', 'tasa_conversion']
        return cat_perf.round(4)
    
    @staticmethod
    def plot_vertical_performance(df: pd.DataFrame, figsize: Tuple = (16, 8)) -> None:
        """
        Genera gráfico de performance por vertical con diseño moderno y profesional.
        
        PREGUNTA: ¿Cómo varía la performance entre las diferentes verticales?
        """
        vertical = AnalisisCategoria.get_vertical_performance(df)
        
        # Configuración de estilo
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor='white')
        fig.suptitle('¿Cómo varía la performance entre las diferentes verticales?', 
                     fontsize=16, fontweight='bold', color='#1A252F', y=1.02)
        
        # Paleta de colores profesional
        color_gmv = '#4A90D9'       # Azul profesional
        color_conv = '#50C878'       # Verde esmeralda
        
        # Ordenar por GMV
        vertical_sorted = vertical.sort_values('gmv_total', ascending=True)
        
        # ═══════════════════════════════════════════════════════════
        # Gráfico 1: GMV por vertical
        # ═══════════════════════════════════════════════════════════
        ax1 = axes[0]
        ax1.set_facecolor('#FAFAFA')
        
        # Crear gradiente de colores basado en valores
        gmv_normalized = (vertical_sorted['gmv_total'] - vertical_sorted['gmv_total'].min()) / \
                        (vertical_sorted['gmv_total'].max() - vertical_sorted['gmv_total'].min())
        colors_gmv = [plt.cm.Blues(0.4 + 0.5 * val) for val in gmv_normalized]
        
        bars1 = ax1.barh(vertical_sorted.index, vertical_sorted['gmv_total'], 
                        color=colors_gmv, edgecolor='white', linewidth=1.2, height=0.7)
        
        # Agregar valores en las barras
        max_gmv = vertical_sorted['gmv_total'].max()
        for bar, value in zip(bars1, vertical_sorted['gmv_total']):
            width = bar.get_width()
            # Posicionar texto dentro o fuera según el tamaño
            if width > max_gmv * 0.3:
                ax1.text(width - max_gmv * 0.02, bar.get_y() + bar.get_height()/2,
                        f'${value:,.0f}', ha='right', va='center',
                        fontsize=9, fontweight='bold', color='white')
            else:
                ax1.text(width + max_gmv * 0.01, bar.get_y() + bar.get_height()/2,
                        f'${value:,.0f}', ha='left', va='center',
                        fontsize=9, fontweight='medium', color='#2C3E50')
        
        ax1.set_xlabel('GMV Total ($)', fontsize=11, fontweight='medium', 
                      labelpad=10, color='#2C3E50')
        ax1.set_title('💰 GMV Total', fontsize=13, fontweight='bold', 
                     pad=15, color='#2C3E50')
        
        # Estilo de ejes
        ax1.tick_params(axis='y', labelsize=10, colors='#34495E')
        ax1.tick_params(axis='x', labelsize=9, colors='#34495E')
        ax1.xaxis.grid(True, linestyle='-', alpha=0.3, color='#BDC3C7')
        ax1.yaxis.grid(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_color('#BDC3C7')
        ax1.spines['bottom'].set_color('#BDC3C7')
        
        # Marcar el mejor
        best_gmv_idx = vertical_sorted['gmv_total'].idxmax()
        best_gmv_pos = list(vertical_sorted.index).index(best_gmv_idx)
        ax1.scatter(vertical_sorted.loc[best_gmv_idx, 'gmv_total'] * 1.08, best_gmv_pos,
                   marker='*', s=200, color='#F39C12', zorder=5, edgecolors='white', linewidths=0.5)
        
        # ═══════════════════════════════════════════════════════════
        # Gráfico 2: Tasa de conversión por vertical
        # ═══════════════════════════════════════════════════════════
        ax2 = axes[1]
        ax2.set_facecolor('#FAFAFA')
        
        # Crear gradiente de colores basado en conversión
        conv_values = vertical_sorted['tasa_conversion'] * 100
        conv_normalized = (conv_values - conv_values.min()) / (conv_values.max() - conv_values.min())
        colors_conv = [plt.cm.Greens(0.4 + 0.5 * val) for val in conv_normalized]
        
        bars2 = ax2.barh(vertical_sorted.index, conv_values, 
                        color=colors_conv, edgecolor='white', linewidth=1.2, height=0.7)
        
        # Agregar valores en las barras
        max_conv = conv_values.max()
        for bar, value in zip(bars2, conv_values):
            width = bar.get_width()
            if width > max_conv * 0.3:
                ax2.text(width - max_conv * 0.02, bar.get_y() + bar.get_height()/2,
                        f'{value:.1f}%', ha='right', va='center',
                        fontsize=9, fontweight='bold', color='white')
            else:
                ax2.text(width + max_conv * 0.01, bar.get_y() + bar.get_height()/2,
                        f'{value:.1f}%', ha='left', va='center',
                        fontsize=9, fontweight='medium', color='#2C3E50')
        
        # Línea de promedio
        avg_conv = conv_values.mean()
        ax2.axvline(x=avg_conv, color='#E74C3C', linestyle='--', 
                   alpha=0.7, linewidth=2, zorder=2)
        ax2.text(avg_conv + max_conv * 0.02, len(vertical_sorted) - 0.5, 
                f'Prom: {avg_conv:.1f}%', fontsize=9, color='#E74C3C', 
                fontweight='medium', va='center')
        
        ax2.set_xlabel('Tasa de Conversión (%)', fontsize=11, fontweight='medium', 
                      labelpad=10, color='#2C3E50')
        ax2.set_title('🎯 Tasa de Conversión', fontsize=13, fontweight='bold', 
                     pad=15, color='#2C3E50')
        
        # Estilo de ejes
        ax2.tick_params(axis='y', labelsize=10, colors='#34495E')
        ax2.tick_params(axis='x', labelsize=9, colors='#34495E')
        ax2.xaxis.grid(True, linestyle='-', alpha=0.3, color='#BDC3C7')
        ax2.yaxis.grid(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_color('#BDC3C7')
        ax2.spines['bottom'].set_color('#BDC3C7')
        
        # Marcar el mejor
        best_conv_idx = vertical_sorted['tasa_conversion'].idxmax()
        best_conv_pos = list(vertical_sorted.index).index(best_conv_idx)
        ax2.scatter(vertical_sorted.loc[best_conv_idx, 'tasa_conversion'] * 100 * 1.08, best_conv_pos,
                   marker='*', s=200, color='#F39C12', zorder=5, edgecolors='white', linewidths=0.5)
        
        # Ocultar labels del eje Y en el segundo gráfico (ya están en el primero)
        ax2.set_yticklabels([])
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        
        # Imprimir resumen
        top_gmv = vertical['gmv_total'].idxmax()
        top_conv = vertical['tasa_conversion'].idxmax()
        worst_conv = vertical['tasa_conversion'].idxmin()
        print()
        print("=" * 60)
        print("📦 PERFORMANCE POR VERTICAL")
        print("=" * 60)
        print(f"Total verticales: {len(vertical)}")
        print("-" * 60)
        print(f"💰 Mayor GMV:           {top_gmv} (${vertical.loc[top_gmv, 'gmv_total']:,.0f})")
        print(f"🏆 Mayor conversión:    {top_conv} ({vertical.loc[top_conv, 'tasa_conversion']:.1%})")
        print(f"📉 Menor conversión:    {worst_conv} ({vertical.loc[worst_conv, 'tasa_conversion']:.1%})")
        print("-" * 60)
        print("📋 Top 5 por GMV:")
        for i, (idx, row) in enumerate(vertical.head(5).iterrows(), 1):
            print(f"   {i}. {idx}: ${row['gmv_total']:,.0f} | Conv: {row['tasa_conversion']:.1%}")
        print("=" * 60)
    
    @staticmethod
    def plot_top_domains(df: pd.DataFrame, top_n: int = 20, figsize: Tuple = (18, 10)) -> None:
        """
        Genera gráfico de top dominios por GMV con diseño de dos paneles (GMV + Conversión).
        
        PREGUNTA: ¿Cuáles son los dominios con mayor GMV y cómo es su conversión?
        """
        domain = AnalisisCategoria.get_domain_performance(df, top_n)
        
        # Configuración de estilo
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor='white')
        fig.suptitle(f'¿Cuáles son los top {top_n} dominios con mayor GMV y cómo es su conversión?', 
                     fontsize=16, fontweight='bold', color='#1A252F', y=1.02)
        
        # Paleta de colores profesional
        color_gmv = '#4A90D9'       # Azul profesional
        color_conv = '#50C878'       # Verde esmeralda
        
        # Ordenar por GMV
        domain_sorted = domain.sort_values('gmv_total', ascending=True)
        
        # Etiquetas truncadas para el eje Y
        y_labels = [name[:30] + '...' if len(name) > 30 else name for name in domain_sorted.index]
        y_positions = range(len(domain_sorted))
        
        # ═══════════════════════════════════════════════════════════
        # Gráfico 1: GMV por dominio
        # ═══════════════════════════════════════════════════════════
        ax1 = axes[0]
        ax1.set_facecolor('#FAFAFA')
        
        # Crear gradiente de colores basado en valores
        gmv_normalized = (domain_sorted['gmv_total'] - domain_sorted['gmv_total'].min()) / \
                        (domain_sorted['gmv_total'].max() - domain_sorted['gmv_total'].min())
        colors_gmv = [plt.cm.Blues(0.4 + 0.5 * val) for val in gmv_normalized]
        
        bars1 = ax1.barh(y_positions, domain_sorted['gmv_total'], 
                        color=colors_gmv, edgecolor='white', linewidth=1.2, height=0.7)
        
        # Agregar valores en las barras
        max_gmv = domain_sorted['gmv_total'].max()
        for i, (bar, value) in enumerate(zip(bars1, domain_sorted['gmv_total'])):
            width = bar.get_width()
            if width > max_gmv * 0.3:
                ax1.text(width - max_gmv * 0.02, i,
                        f'${value:,.0f}', ha='right', va='center',
                        fontsize=8, fontweight='bold', color='white')
            else:
                ax1.text(width + max_gmv * 0.01, i,
                        f'${value:,.0f}', ha='left', va='center',
                        fontsize=8, fontweight='medium', color='#2C3E50')
        
        ax1.set_xlabel('GMV Total ($)', fontsize=11, fontweight='medium', 
                      labelpad=10, color='#2C3E50')
        ax1.set_title('💰 GMV Total', fontsize=13, fontweight='bold', 
                     pad=15, color='#2C3E50')
        
        # Configurar etiquetas Y
        ax1.set_yticks(y_positions)
        ax1.set_yticklabels(y_labels, fontsize=9, color='#34495E')
        
        # Estilo de ejes
        ax1.tick_params(axis='x', labelsize=9, colors='#34495E')
        ax1.xaxis.grid(True, linestyle='-', alpha=0.3, color='#BDC3C7')
        ax1.yaxis.grid(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_color('#BDC3C7')
        ax1.spines['bottom'].set_color('#BDC3C7')
        
        # Marcar el mejor
        best_gmv_pos = len(domain_sorted) - 1
        ax1.scatter(domain_sorted.iloc[best_gmv_pos]['gmv_total'] * 1.08, best_gmv_pos,
                   marker='*', s=200, color='#F39C12', zorder=5, edgecolors='white', linewidths=0.5)
        
        # ═══════════════════════════════════════════════════════════
        # Gráfico 2: Tasa de conversión por dominio
        # ═══════════════════════════════════════════════════════════
        ax2 = axes[1]
        ax2.set_facecolor('#FAFAFA')
        
        # Crear gradiente de colores basado en conversión
        conv_values = domain_sorted['tasa_conversion'] * 100
        conv_normalized = (conv_values - conv_values.min()) / (conv_values.max() - conv_values.min() + 0.001)
        colors_conv = [plt.cm.Greens(0.4 + 0.5 * val) for val in conv_normalized]
        
        bars2 = ax2.barh(y_positions, conv_values, 
                        color=colors_conv, edgecolor='white', linewidth=1.2, height=0.7)
        
        # Agregar valores en las barras
        max_conv = conv_values.max()
        for i, (bar, value) in enumerate(zip(bars2, conv_values)):
            width = bar.get_width()
            if width > max_conv * 0.3:
                ax2.text(width - max_conv * 0.02, i,
                        f'{value:.1f}%', ha='right', va='center',
                        fontsize=8, fontweight='bold', color='white')
            else:
                ax2.text(width + max_conv * 0.01, i,
                        f'{value:.1f}%', ha='left', va='center',
                        fontsize=8, fontweight='medium', color='#2C3E50')
        
        # Línea de promedio
        avg_conv = conv_values.mean()
        ax2.axvline(x=avg_conv, color='#E74C3C', linestyle='--', 
                   alpha=0.7, linewidth=2, zorder=2)
        ax2.text(avg_conv + max_conv * 0.02, len(domain_sorted) - 0.5, 
                f'Prom: {avg_conv:.1f}%', fontsize=9, color='#E74C3C', 
                fontweight='medium', va='center')
        
        ax2.set_xlabel('Tasa de Conversión (%)', fontsize=11, fontweight='medium', 
                      labelpad=10, color='#2C3E50')
        ax2.set_title('🎯 Tasa de Conversión', fontsize=13, fontweight='bold', 
                     pad=15, color='#2C3E50')
        
        # Estilo de ejes
        ax2.set_yticks(y_positions)
        ax2.set_yticklabels([])  # Ocultar labels (ya están en el primer gráfico)
        ax2.tick_params(axis='x', labelsize=9, colors='#34495E')
        ax2.xaxis.grid(True, linestyle='-', alpha=0.3, color='#BDC3C7')
        ax2.yaxis.grid(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_color('#BDC3C7')
        ax2.spines['bottom'].set_color('#BDC3C7')
        
        # Marcar el mejor en conversión
        best_conv_idx = domain_sorted['tasa_conversion'].idxmax()
        best_conv_pos = list(domain_sorted.index).index(best_conv_idx)
        ax2.scatter(domain_sorted.loc[best_conv_idx, 'tasa_conversion'] * 100 * 1.08, best_conv_pos,
                   marker='*', s=200, color='#F39C12', zorder=5, edgecolors='white', linewidths=0.5)
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        
        # Imprimir resumen
        total_gmv = df['SOLD_AMOUNT'].sum()
        top_gmv = domain['gmv_total'].sum()
        low_conv = (domain['tasa_conversion'] < 0.5).sum()
        high_conv = (domain['tasa_conversion'] >= 0.5).sum()
        print()
        print("=" * 60)
        print(f"🏆 TOP {top_n} DOMINIOS POR GMV")
        print("=" * 60)
        print(f"GMV de top {top_n}: ${top_gmv:,.0f} ({top_gmv/total_gmv:.1%} del total)")
        print(f"🟢 Con buena conversión (≥50%): {high_conv}")
        print(f"🔴 Con baja conversión (<50%):  {low_conv}")
        print("-" * 60)
        print("📋 Detalle:")
        for i, (idx, row) in enumerate(domain.head(10).iterrows(), 1):
            emoji = "🟢" if row['tasa_conversion'] >= 0.5 else "🔴"
            print(f"   {i}. {idx[:40]}: ${row['gmv_total']:,.0f} | {emoji} {row['tasa_conversion']:.1%}")
        if top_n > 10:
            print(f"   ... y {top_n - 10} más")
        print("=" * 60)
    
    @staticmethod
    def plot_pareto_curve(df: pd.DataFrame, figsize: Tuple = (10, 6)) -> None:
        """
        Genera curva de Pareto de concentración de GMV.
        
        PREGUNTA: ¿Cuántos dominios concentran la mayor parte del GMV?
        """
        pareto = AnalisisCategoria.get_pareto_analysis(df)
        cumsum = pareto['gmv_cumsum']
        
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle('¿Cuántos dominios concentran la mayor parte del GMV?', 
                    fontsize=14, fontweight='bold', color='#1A252F', y=1.02)
        
        x = range(1, len(cumsum) + 1)
        ax.plot(x, cumsum.values * 100, color='#3498db', linewidth=2)
        ax.axhline(y=80, color='red', linestyle='--', label='80% del GMV')
        ax.axvline(x=pareto['dominios_80_gmv'], color='red', linestyle='--', alpha=0.5)
        
        ax.fill_between(x, cumsum.values * 100, alpha=0.3)
        ax.set_xlabel('Número de Dominios (ordenados por GMV)')
        ax.set_ylabel('% Acumulado del GMV')
        ax.set_title(f'Curva de Pareto: {pareto["dominios_80_gmv"]} dominios ({pareto["pct_dominios_80_gmv"]:.1f}%) generan 80% del GMV')
        ax.legend()
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        
        # Imprimir resumen
        print()
        print("=" * 60)
        print("📊 ANÁLISIS DE PARETO - CONCENTRACIÓN DE GMV")
        print("=" * 60)
        print(f"Total de dominios: {pareto['total_dominios']:,}")
        print("-" * 60)
        print(f"📈 80% del GMV:  {pareto['dominios_80_gmv']} dominios ({pareto['pct_dominios_80_gmv']:.1f}% del total)")
        print(f"📈 90% del GMV:  {pareto['dominios_90_gmv']} dominios ({pareto['pct_dominios_90_gmv']:.1f}% del total)")
        print(f"🏆 Top 10:       {pareto['top10_gmv_pct']:.1f}% del GMV total")
        print("-" * 60)
        print("💡 INSIGHT: Alta concentración indica dependencia de pocos dominios")
        print("=" * 60)
    
    @staticmethod
    def plot_sellout_vs_gmv(df: pd.DataFrame, figsize: Tuple = (12, 10)) -> None:
        """
        Genera scatter plot estilo BCG de tasa de agotamiento vs GMV por categoría.
        
        PREGUNTA: ¿Cuáles son las categorías 'estrella' con alto sellout y alto GMV?
        """
        cat_perf = AnalisisCategoria.get_sellout_vs_gmv_by_category(df)
        
        # Identificar cuadrantes
        median_sellout = cat_perf['tasa_sellout'].median()
        median_gmv = cat_perf['gmv_total'].median()
        
        # Asignar cuadrante a cada categoría
        def get_quadrant(row):
            if row['tasa_sellout'] >= median_sellout and row['gmv_total'] >= median_gmv:
                return 'Estrellas ⭐'
            elif row['tasa_sellout'] < median_sellout and row['gmv_total'] >= median_gmv:
                return 'Vacas 🐄'
            elif row['tasa_sellout'] >= median_sellout and row['gmv_total'] < median_gmv:
                return 'Interrogantes ❓'
            else:
                return 'Perros 🐕'
        
        cat_perf = cat_perf.copy()
        cat_perf['quadrant'] = cat_perf.apply(get_quadrant, axis=1)
        
        # Colores por cuadrante
        quadrant_colors = {
            'Estrellas ⭐': COLORS['success'],
            'Vacas 🐄': COLORS['primary'],
            'Interrogantes ❓': COLORS['warning'],
            'Perros 🐕': COLORS['danger']
        }
        
        # Imprimir resumen primero
        stars = cat_perf[cat_perf['quadrant'] == 'Estrellas ⭐']
        cows = cat_perf[cat_perf['quadrant'] == 'Vacas 🐄']
        questions = cat_perf[cat_perf['quadrant'] == 'Interrogantes ❓']
        dogs = cat_perf[cat_perf['quadrant'] == 'Perros 🐕']
        
        print()
        print("=" * 60)
        print("🎯 MATRIZ BCG: SELLOUT vs GMV POR CATEGORÍA")
        print("=" * 60)
        print(f"Total categorías: {len(cat_perf)}")
        print("-" * 60)
        print(f"⭐ Estrellas (alto sellout + alto GMV):    {len(stars)} categorías")
        print(f"🐄 Vacas (bajo sellout + alto GMV):        {len(cows)} categorías")
        print(f"❓ Interrogantes (alto sellout + bajo GMV): {len(questions)} categorías")
        print(f"🐕 Perros (bajo sellout + bajo GMV):       {len(dogs)} categorías")
        print("-" * 60)
        if len(stars) > 0:
            print("⭐ Top Estrellas:")
            for idx in stars.nlargest(3, 'gmv_total').index:
                print(f"   - {idx}: ${cat_perf.loc[idx, 'gmv_total']:,.0f} | Sellout: {cat_perf.loc[idx, 'tasa_sellout']:.1%}")
        print("=" * 60)
        print()
        
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle('¿Cuáles son las categorías "estrella" con alto sellout y alto GMV?', 
                    fontsize=14, fontweight='bold', color='#1A252F', y=0.98)
        
        # Calcular límites
        xlim_max = cat_perf['tasa_sellout'].max() * 100 * 1.1
        ylim_max = cat_perf['gmv_total'].max() * 1.1
        
        # Sombrear cuadrantes con colores suaves
        ax.axhspan(median_gmv, ylim_max, xmin=0, xmax=median_sellout*100/xlim_max, 
                  alpha=0.08, color=COLORS['primary'])  # Vacas
        ax.axhspan(median_gmv, ylim_max, xmin=median_sellout*100/xlim_max, xmax=1, 
                  alpha=0.08, color=COLORS['success'])  # Estrellas
        ax.axhspan(0, median_gmv, xmin=0, xmax=median_sellout*100/xlim_max, 
                  alpha=0.08, color=COLORS['danger'])  # Perros
        ax.axhspan(0, median_gmv, xmin=median_sellout*100/xlim_max, xmax=1, 
                  alpha=0.08, color=COLORS['warning'])  # Interrogantes
        
        # Scatter plot por cuadrante
        for quadrant, color in quadrant_colors.items():
            mask = cat_perf['quadrant'] == quadrant
            ax.scatter(cat_perf.loc[mask, 'tasa_sellout'] * 100, 
                      cat_perf.loc[mask, 'gmv_total'],
                      s=cat_perf.loc[mask, 'tasa_conversion'] * 800 + 50,
                      alpha=0.7, c=color, label=quadrant, edgecolor='white', linewidth=1.5)
        
        # Agregar etiquetas con mejor posicionamiento
        for idx, row in cat_perf.iterrows():
            # Truncar nombres largos
            label = idx[:20] + '...' if len(idx) > 20 else idx
            ax.annotate(label, (row['tasa_sellout'] * 100 + 0.5, row['gmv_total']),
                       fontsize=8, alpha=0.85, fontweight='medium')
        
        # Líneas de cuadrantes
        ax.axhline(y=median_gmv, color=COLORS['dark'], linestyle='--', alpha=0.5, linewidth=1.5)
        ax.axvline(x=median_sellout * 100, color=COLORS['dark'], linestyle='--', alpha=0.5, linewidth=1.5)
        
        # Etiquetas de cuadrantes
        ax.text(median_sellout * 50, ylim_max * 0.95, '🐄 VACAS\n(Optimizar)', 
               ha='center', va='top', fontsize=10, alpha=0.6, fontweight='bold')
        ax.text((median_sellout * 100 + xlim_max) / 2, ylim_max * 0.95, '⭐ ESTRELLAS\n(Mantener)', 
               ha='center', va='top', fontsize=10, alpha=0.6, fontweight='bold')
        ax.text(median_sellout * 50, median_gmv * 0.15, '🐕 PERROS\n(Revisar)', 
               ha='center', va='bottom', fontsize=10, alpha=0.6, fontweight='bold')
        ax.text((median_sellout * 100 + xlim_max) / 2, median_gmv * 0.15, '❓ INTERROGANTES\n(Potencial)', 
               ha='center', va='bottom', fontsize=10, alpha=0.6, fontweight='bold')
        
        ax.set_xlabel('Tasa de Sellout (%)', fontweight='medium')
        ax.set_ylabel('GMV Total ($)', fontweight='medium')
        ax.set_title('Matriz BCG: Sellout vs GMV por Categoría\n(Tamaño de burbuja = Tasa de Conversión)', pad=15)
        ax.legend(loc='upper left', framealpha=0.9, title='Cuadrante')
        ax.set_xlim(0, xlim_max)
        ax.set_ylim(0, ylim_max)
        
        # Formatear eje Y con K/M
        ax.yaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, p: f'${x/1e6:.1f}M' if x >= 1e6 else f'${x/1e3:.0f}K'))
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)


# =============================================================================
# TEMÁTICA 4: PRICING, GMV Y VELOCIDAD (Parte 1)
# =============================================================================
# Clases: AnalisisPricing, AnalisisVelocidad
# Preguntas: Ticket, precio vs performance, métricas de velocidad
# =============================================================================

class AnalisisPricing:
    """Métodos para análisis de pricing y GMV."""
    
    @staticmethod
    def get_ticket_by_category(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula el ticket promedio por categoría.
        """
        df_con_ventas = df[df['has_sales']]
        ticket = df_con_ventas.groupby('VERTICAL').agg({
            'avg_ticket': ['mean', 'median', 'std', 'count']
        })
        ticket.columns = ['ticket_promedio', 'ticket_mediano', 'ticket_std', 'num_ofertas']
        return ticket.sort_values('ticket_promedio', ascending=False).round(2)
    
    @staticmethod
    def get_ticket_performance(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula la performance por rangos de ticket.
        """
        df_con_ventas = df[df['has_sales']].copy()
        
        # Crear buckets de ticket
        ticket_bins = [0, 2, 5, 10, 20, 50, 100, float('inf')]
        ticket_labels = ['$0-2', '$2-5', '$5-10', '$10-20', '$20-50', '$50-100', '>$100']
        df_con_ventas['ticket_bucket'] = pd.cut(df_con_ventas['avg_ticket'], 
                                                 bins=ticket_bins, labels=ticket_labels)
        
        ticket_perf = df_con_ventas.groupby('ticket_bucket').agg({
            'OFFER_START_DATE': 'count',
            'is_sold_out': 'mean',
            'sell_through_rate': 'mean',
            'gmv_per_hour': 'mean',
            'SOLD_AMOUNT': 'sum'
        })
        ticket_perf.columns = ['num_ofertas', 'tasa_sellout', 'sell_through_rate',
                               'gmv_por_hora', 'gmv_total']
        return ticket_perf.round(4)
    
    @staticmethod
    def plot_ticket_by_vertical(df: pd.DataFrame, figsize: Tuple = (14, 7)) -> None:
        """
        Genera violin plot con strip plot de ticket promedio por vertical.
        
        PREGUNTA: ¿Cómo varía el ticket promedio entre verticales?
        """
        df_con_ventas = df[df['has_sales']].copy()
        ticket_stats = AnalisisPricing.get_ticket_by_category(df)
        
        # Filtrar outliers extremos para mejor visualización
        q99 = df_con_ventas['avg_ticket'].quantile(0.99)
        df_plot = df_con_ventas[df_con_ventas['avg_ticket'] <= q99]
        
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle('¿Cómo varía el ticket promedio entre verticales?', 
                    fontsize=14, fontweight='bold', color='#1A252F', y=1.02)
        
        # Ordenar verticales por mediana de ticket
        order = df_plot.groupby('VERTICAL')['avg_ticket'].median().sort_values(ascending=False).index
        
        # Violin plot base (muestra la distribución completa)
        sns.violinplot(data=df_plot, x='VERTICAL', y='avg_ticket', order=order, ax=ax,
                      inner=None, alpha=0.3, palette=COLOR_PALETTE, cut=0)
        
        # Boxplot encima (muestra cuartiles)
        sns.boxplot(data=df_plot, x='VERTICAL', y='avg_ticket', order=order, ax=ax,
                   width=0.15, boxprops=dict(alpha=0.8), 
                   whiskerprops=dict(color=COLORS['dark']),
                   medianprops=dict(color=COLORS['danger'], linewidth=2),
                   fliersize=0)  # Sin outliers
        
        # Strip plot (puntos individuales) - muestra cada dato
        sns.stripplot(data=df_plot, x='VERTICAL', y='avg_ticket', order=order, ax=ax,
                     size=2, alpha=0.2, color=COLORS['dark'], jitter=0.25)
        
        # Agregar línea de mediana global
        global_median = df_plot['avg_ticket'].median()
        ax.axhline(y=global_median, color=COLORS['secondary'], linestyle='--', 
                  linewidth=2, alpha=0.7, label=f'Mediana global: ${global_median:.2f}')
        
        # Agregar anotaciones de mediana por vertical
        for i, vertical in enumerate(order):
            median_val = df_plot[df_plot['VERTICAL'] == vertical]['avg_ticket'].median()
            ax.annotate(f'${median_val:.0f}', (i, median_val), 
                       textcoords="offset points", xytext=(0, 10),
                       ha='center', fontsize=9, fontweight='bold', color=COLORS['dark'])
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_xlabel('Vertical', fontweight='medium')
        ax.set_ylabel('Ticket Promedio ($)', fontweight='medium')
        ax.set_title('Distribución de Ticket Promedio por Vertical\n(Violin + Box + Strip Plot)', pad=15)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        
        # Imprimir resumen
        top_ticket = ticket_stats['ticket_promedio'].idxmax()
        low_ticket = ticket_stats['ticket_promedio'].idxmin()
        print()
        print("=" * 60)
        print("🎫 TICKET PROMEDIO POR VERTICAL")
        print("=" * 60)
        print(f"Ofertas con ventas analizadas: {len(df_con_ventas):,}")
        print("-" * 60)
        print(f"{'Vertical':<20} {'Promedio':>12} {'Mediana':>12} {'Ofertas':>10}")
        print("-" * 60)
        for idx, row in ticket_stats.iterrows():
            print(f"{str(idx)[:20]:<20} ${row['ticket_promedio']:>10,.2f} ${row['ticket_mediano']:>10,.2f} {int(row['num_ofertas']):>10,}")
        print("-" * 60)
        print(f"💰 Mayor ticket: {top_ticket} (${ticket_stats.loc[top_ticket, 'ticket_promedio']:.2f})")
        print(f"📉 Menor ticket: {low_ticket} (${ticket_stats.loc[low_ticket, 'ticket_promedio']:.2f})")
        print("=" * 60)
    
    @staticmethod
    def get_ticket_by_category_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calcula el ticket promedio por categoría, separando ofertas exitosas vs zombies.
        
        Returns:
            Tuple con (df_exitosas, df_zombies)
        """
        # Ofertas exitosas (con ventas)
        df_exitosas = df[df['has_sales']]
        ticket_exitosas = df_exitosas.groupby('VERTICAL').agg({
            'avg_ticket': ['mean', 'median', 'std', 'count']
        })
        ticket_exitosas.columns = ['ticket_promedio', 'ticket_mediano', 'ticket_std', 'num_ofertas']
        ticket_exitosas = ticket_exitosas.sort_values('ticket_promedio', ascending=False).round(2)
        
        # Ofertas zombies (sin ventas) - solo contamos por vertical (no hay precio disponible)
        df_zombies = df[~df['has_sales']]
        ticket_zombies = df_zombies.groupby('VERTICAL').size().reset_index(name='num_ofertas')
        ticket_zombies = ticket_zombies.set_index('VERTICAL').sort_values('num_ofertas', ascending=False)
        
        return ticket_exitosas, ticket_zombies
    
    @staticmethod
    def plot_ticket_by_vertical_split(df: pd.DataFrame, figsize: Tuple = (14, 6)) -> None:
        """
        Genera gráfico de barras comparando ofertas exitosas vs zombies por vertical.
        Muestra ticket mediano para exitosas y cantidad para zombies (no hay precio disponible para zombies).
        
        PREGUNTA: ¿Qué verticales tienen más ofertas exitosas vs zombies?
        """
        import numpy as np
        
        df_exitosas = df[df['has_sales']].copy()
        df_zombies = df[~df['has_sales']].copy()
        
        # Calcular medianas por vertical (solo exitosas tienen precio disponible)
        median_exitosas = df_exitosas.groupby('VERTICAL')['avg_ticket'].median()
        
        # Contar ofertas por vertical
        count_exitosas = df_exitosas.groupby('VERTICAL').size()
        count_zombies = df_zombies.groupby('VERTICAL').size()
        
        # Usar todas las verticales y ordenar por mediana de exitosas
        all_verticals = median_exitosas.sort_values(ascending=True).index.tolist()
        
        # Preparar datos para el gráfico
        y_pos = np.arange(len(all_verticals))
        bar_height = 0.35
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Colores
        color_exitosas = '#27ae60'  # Verde
        color_zombies = '#e74c3c'   # Rojo
        
        # Panel 1: Ticket mediano de ofertas exitosas
        bars_exitosas = axes[0].barh(y_pos, 
                                [median_exitosas.get(v, 0) for v in all_verticals],
                                bar_height * 2, 
                                label=f'Exitosas (n={len(df_exitosas):,})', 
                                color=color_exitosas, 
                                alpha=0.85,
                                edgecolor='white',
                                linewidth=1)
        
        # Agregar valores en las barras
        for i, v in enumerate(all_verticals):
            val_exit = median_exitosas.get(v, 0)
            if val_exit > 0:
                axes[0].text(val_exit + 0.5, i, f'${val_exit:.1f}', 
                       va='center', ha='left', fontsize=9, fontweight='bold', color=color_exitosas)
        
        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels(all_verticals, fontsize=10)
        axes[0].set_xlabel('Ticket Mediano ($)', fontsize=11, fontweight='medium')
        axes[0].set_title('🟢 Ofertas Exitosas - Ticket por Vertical', 
                    fontsize=12, fontweight='bold', pad=15, color='#1A252F')
        axes[0].xaxis.grid(True, linestyle='--', alpha=0.3)
        axes[0].set_axisbelow(True)
        
        # Panel 2: Conteo de zombies por vertical
        bars_zombies = axes[1].barh(y_pos, 
                               [count_zombies.get(v, 0) for v in all_verticals],
                               bar_height * 2, 
                               label=f'Zombies (n={len(df_zombies):,})', 
                               color=color_zombies, 
                               alpha=0.85,
                               edgecolor='white',
                               linewidth=1)
        
        for i, v in enumerate(all_verticals):
            cnt_zomb = count_zombies.get(v, 0)
            if cnt_zomb > 0:
                axes[1].text(cnt_zomb + 0.5, i, f'{cnt_zomb:,}', 
                       va='center', ha='left', fontsize=9, fontweight='bold', color=color_zombies)
        
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(all_verticals, fontsize=10)
        axes[1].set_xlabel('Cantidad de Ofertas', fontsize=11, fontweight='medium')
        axes[1].set_title('🔴 Ofertas Zombies - Cantidad por Vertical', 
                    fontsize=12, fontweight='bold', pad=15, color='#1A252F')
        axes[1].xaxis.grid(True, linestyle='--', alpha=0.3)
        axes[1].set_axisbelow(True)
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        
        # Resumen simple
        global_median_exit = df_exitosas['avg_ticket'].median()
        
        print()
        print("=" * 70)
        print("📊 RESUMEN: OFERTAS POR VERTICAL")
        print("=" * 70)
        print(f"   🟢 Ofertas Exitosas: {len(df_exitosas):,}  (ticket mediano: ${global_median_exit:.2f})")
        print(f"   🔴 Ofertas Zombies:  {len(df_zombies):,}  (sin precio disponible)")
        print()
        print("   ℹ️  Nota: No hay columna ORIGINAL_PRICE en el dataset para comparar precios de zombies")
        print("=" * 70)
    
    @staticmethod
    def _legacy_plot_ticket_by_vertical_split(df: pd.DataFrame, figsize: Tuple = (16, 7)) -> None:
        """
        [LEGACY] Genera violin plots comparando ofertas exitosas vs zombies por vertical.
        """
        df_exitosas = df[df['has_sales']].copy()
        df_zombies = df[~df['has_sales']].copy()
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # --- Panel izquierdo: Ofertas Exitosas ---
        if len(df_exitosas) > 0:
            q99_exitosas = df_exitosas['avg_ticket'].quantile(0.99)
            df_plot_exitosas = df_exitosas[df_exitosas['avg_ticket'] <= q99_exitosas]
            
            order_exitosas = df_plot_exitosas.groupby('VERTICAL')['avg_ticket'].median().sort_values(ascending=False).index
            
            sns.violinplot(data=df_plot_exitosas, x='VERTICAL', y='avg_ticket', order=order_exitosas, ax=axes[0],
                          inner=None, alpha=0.3, palette=COLOR_PALETTE, cut=0)
            sns.boxplot(data=df_plot_exitosas, x='VERTICAL', y='avg_ticket', order=order_exitosas, ax=axes[0],
                       width=0.15, boxprops=dict(alpha=0.8), 
                       whiskerprops=dict(color=COLORS['dark']),
                       medianprops=dict(color=COLORS['success'], linewidth=2),
                       fliersize=0)
            
            global_median_exitosas = df_plot_exitosas['avg_ticket'].median()
            axes[0].axhline(y=global_median_exitosas, color=COLORS['success'], linestyle='--', 
                          linewidth=2, alpha=0.7, label=f'Mediana: ${global_median_exitosas:.2f}')
            
            axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
            axes[0].set_xlabel('Vertical', fontweight='medium')
            axes[0].set_ylabel('Ticket Promedio ($)', fontweight='medium')
            axes[0].set_title(f'🟢 OFERTAS EXITOSAS (n={len(df_exitosas):,})\nDistribución de Ticket por Vertical', 
                            pad=15, fontsize=12, fontweight='bold', color=COLORS['success'])
            axes[0].legend(loc='upper right')
        
        # --- Panel derecho: Ofertas Zombies (solo conteo, no hay precio disponible) ---
        if len(df_zombies) > 0:
            count_zombies = df_zombies.groupby('VERTICAL').size().sort_values(ascending=False)
            
            bars = axes[1].barh(range(len(count_zombies)), count_zombies.values, color=COLORS['danger'], alpha=0.7)
            axes[1].set_yticks(range(len(count_zombies)))
            axes[1].set_yticklabels(count_zombies.index)
            axes[1].set_xlabel('Cantidad de Ofertas', fontweight='medium')
            axes[1].set_title(f'🔴 OFERTAS ZOMBIES (n={len(df_zombies):,})\nCantidad por Vertical (sin precio disponible)', 
                            pad=15, fontsize=12, fontweight='bold', color=COLORS['danger'])
            
            # Agregar valores en las barras
            for i, (idx, val) in enumerate(count_zombies.items()):
                axes[1].text(val + 10, i, f'{val:,}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        
        # Imprimir resumen comparativo
        ticket_exitosas, ticket_zombies = AnalisisPricing.get_ticket_by_category_split(df)
        
        print()
        print("=" * 100)
        print("🎫 COMPARACIÓN DE PRECIOS: OFERTAS EXITOSAS vs ZOMBIES")
        print("=" * 100)
        print()
        
        # Tabla de ofertas exitosas
        print("🟢 OFERTAS EXITOSAS (con ventas)")
        print("-" * 80)
        print(f"{'Vertical':<20} {'Ticket Prom':>14} {'Ticket Med':>14} {'Std':>12} {'Ofertas':>12}")
        print("-" * 80)
        for idx, row in ticket_exitosas.iterrows():
            print(f"{str(idx)[:20]:<20} ${row['ticket_promedio']:>12,.2f} ${row['ticket_mediano']:>12,.2f} ${row['ticket_std']:>10,.2f} {int(row['num_ofertas']):>12,}")
        print("-" * 80)
        print(f"TOTAL: {int(ticket_exitosas['num_ofertas'].sum()):,} ofertas exitosas")
        print()
        
        # Tabla de ofertas zombies (solo conteo, no hay precio disponible)
        print("🔴 OFERTAS ZOMBIES (sin ventas)")
        print("-" * 80)
        print(f"{'Vertical':<30} {'Ofertas':>12}")
        print("-" * 80)
        for idx, row in ticket_zombies.iterrows():
            print(f"{str(idx)[:30]:<30} {int(row['num_ofertas']):>12,}")
        print("-" * 80)
        print(f"TOTAL: {int(ticket_zombies['num_ofertas'].sum()):,} ofertas zombies")
        print()
        
        # Resumen
        if len(df_exitosas) > 0 and len(df_zombies) > 0:
            median_exitosas = df_exitosas['avg_ticket'].median()
            print("=" * 100)
            print("📊 INSIGHT:")
            print(f"   • Mediana ticket exitosas: ${median_exitosas:.2f}")
            print(f"   • Ofertas zombies: {len(df_zombies):,} (sin precio original disponible en el dataset)")
            print("=" * 100)
    
    @staticmethod
    def plot_ticket_vs_performance(df: pd.DataFrame, figsize: Tuple = (14, 8)) -> None:
        """
        Genera gráfico de ticket vs performance.
        
        PREGUNTA: ¿Cómo afecta el precio a la tasa de éxito de las ofertas?
        """
        ticket_perf = AnalisisPricing.get_ticket_performance(df)
        
        # Colores profesionales
        colors = {
            'sellout': '#00B894',      # Verde menta
            'str': '#0984E3',          # Azul brillante
            'gmv': '#6C5CE7',          # Púrpura
            'ofertas': '#FDCB6E',      # Amarillo dorado
            'highlight': '#E17055',    # Coral para destacar
            'text': '#2D3436',         # Gris oscuro
            'grid': '#DFE6E9'          # Gris claro
        }
        
        fig = plt.figure(figsize=figsize, facecolor='white')
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.35, wspace=0.3)
        
        # Título principal
        fig.suptitle('¿Cómo afecta el precio a la tasa de éxito de las ofertas?', 
                    fontsize=16, fontweight='bold', color=colors['text'], y=0.98)
        
        # === Gráfico 1: Tasa de Sellout (arriba izquierda) ===
        ax1 = fig.add_subplot(gs[0, 0])
        sellout_values = ticket_perf['tasa_sellout'] * 100
        bars1 = ax1.bar(range(len(ticket_perf)), sellout_values, color=colors['sellout'], 
                       edgecolor='white', linewidth=1.5, alpha=0.9)
        
        # Destacar el mejor
        best_sellout_idx = sellout_values.argmax()
        bars1[best_sellout_idx].set_color(colors['highlight'])
        bars1[best_sellout_idx].set_alpha(1.0)
        
        # Etiquetas en barras
        for i, (bar, val) in enumerate(zip(bars1, sellout_values)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold',
                    color=colors['highlight'] if i == best_sellout_idx else colors['text'])
        
        ax1.set_xticks(range(len(ticket_perf)))
        ax1.set_xticklabels(ticket_perf.index, rotation=45, ha='right', fontsize=9)
        ax1.set_ylabel('Tasa de Sellout (%)', fontsize=10, fontweight='bold')
        ax1.set_title('🏆 Tasa de Sellout', fontsize=12, fontweight='bold', color=colors['text'], pad=10)
        ax1.set_ylim(0, max(sellout_values) * 1.15)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.grid(axis='y', alpha=0.3, linestyle='--', color=colors['grid'])
        
        # === Gráfico 2: Sell-Through Rate (arriba centro) ===
        ax2 = fig.add_subplot(gs[0, 1])
        str_values = ticket_perf['sell_through_rate'] * 100
        bars2 = ax2.bar(range(len(ticket_perf)), str_values, color=colors['str'],
                       edgecolor='white', linewidth=1.5, alpha=0.9)
        
        # Destacar el mejor
        best_str_idx = str_values.argmax()
        bars2[best_str_idx].set_color(colors['highlight'])
        bars2[best_str_idx].set_alpha(1.0)
        
        # Etiquetas en barras
        for i, (bar, val) in enumerate(zip(bars2, str_values)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold',
                    color=colors['highlight'] if i == best_str_idx else colors['text'])
        
        ax2.set_xticks(range(len(ticket_perf)))
        ax2.set_xticklabels(ticket_perf.index, rotation=45, ha='right', fontsize=9)
        ax2.set_ylabel('Sell-Through Rate (%)', fontsize=10, fontweight='bold')
        ax2.set_title('📈 Sell-Through Rate', fontsize=12, fontweight='bold', color=colors['text'], pad=10)
        ax2.set_ylim(0, max(str_values) * 1.15)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.grid(axis='y', alpha=0.3, linestyle='--', color=colors['grid'])
        
        # === Gráfico 3: Número de Ofertas (arriba derecha) ===
        ax3 = fig.add_subplot(gs[0, 2])
        ofertas_values = ticket_perf['num_ofertas']
        bars3 = ax3.bar(range(len(ticket_perf)), ofertas_values, color=colors['ofertas'],
                       edgecolor='white', linewidth=1.5, alpha=0.9)
        
        # Destacar el mayor
        most_common_idx = ofertas_values.argmax()
        bars3[most_common_idx].set_color(colors['highlight'])
        bars3[most_common_idx].set_alpha(1.0)
        
        # Etiquetas en barras
        for i, (bar, val) in enumerate(zip(bars3, ofertas_values)):
            label = f'{int(val):,}' if val >= 1000 else str(int(val))
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(ofertas_values)*0.02, 
                    label, ha='center', va='bottom', fontsize=8, fontweight='bold',
                    color=colors['highlight'] if i == most_common_idx else colors['text'])
        
        ax3.set_xticks(range(len(ticket_perf)))
        ax3.set_xticklabels(ticket_perf.index, rotation=45, ha='right', fontsize=9)
        ax3.set_ylabel('Número de Ofertas', fontsize=10, fontweight='bold')
        ax3.set_title('📊 Volumen de Ofertas', fontsize=12, fontweight='bold', color=colors['text'], pad=10)
        ax3.set_ylim(0, max(ofertas_values) * 1.15)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.grid(axis='y', alpha=0.3, linestyle='--', color=colors['grid'])
        
        # === Gráfico 4: GMV Total (abajo izquierda) ===
        ax4 = fig.add_subplot(gs[1, 0])
        gmv_values = ticket_perf['gmv_total'] / 1_000_000  # En millones
        bars4 = ax4.bar(range(len(ticket_perf)), gmv_values, color=colors['gmv'],
                       edgecolor='white', linewidth=1.5, alpha=0.9)
        
        # Destacar el mayor
        best_gmv_idx = gmv_values.argmax()
        bars4[best_gmv_idx].set_color(colors['highlight'])
        bars4[best_gmv_idx].set_alpha(1.0)
        
        # Etiquetas en barras
        for i, (bar, val) in enumerate(zip(bars4, gmv_values)):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(gmv_values)*0.02, 
                    f'${val:.1f}M', ha='center', va='bottom', fontsize=8, fontweight='bold',
                    color=colors['highlight'] if i == best_gmv_idx else colors['text'])
        
        ax4.set_xticks(range(len(ticket_perf)))
        ax4.set_xticklabels(ticket_perf.index, rotation=45, ha='right', fontsize=9)
        ax4.set_ylabel('GMV Total (Millones $)', fontsize=10, fontweight='bold')
        ax4.set_title('💰 GMV Total Generado', fontsize=12, fontweight='bold', color=colors['text'], pad=10)
        ax4.set_ylim(0, max(gmv_values) * 1.15)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.grid(axis='y', alpha=0.3, linestyle='--', color=colors['grid'])
        
        # === Gráfico 5: Comparativo dual - Sellout vs STR (abajo centro-derecha) ===
        ax5 = fig.add_subplot(gs[1, 1:])
        x = np.arange(len(ticket_perf))
        width = 0.35
        
        bars5a = ax5.bar(x - width/2, sellout_values, width, label='Tasa Sellout', 
                        color=colors['sellout'], edgecolor='white', linewidth=1, alpha=0.85)
        bars5b = ax5.bar(x + width/2, str_values, width, label='Sell-Through Rate', 
                        color=colors['str'], edgecolor='white', linewidth=1, alpha=0.85)
        
        ax5.set_xticks(x)
        ax5.set_xticklabels(ticket_perf.index, rotation=45, ha='right', fontsize=10)
        ax5.set_ylabel('Porcentaje (%)', fontsize=10, fontweight='bold')
        ax5.set_title('🔄 Comparativo: Sellout vs Sell-Through Rate', fontsize=12, fontweight='bold', 
                     color=colors['text'], pad=10)
        ax5.legend(loc='upper right', framealpha=0.9, fontsize=9)
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
        ax5.grid(axis='y', alpha=0.3, linestyle='--', color=colors['grid'])
        
        # Añadir línea de tendencia si hay suficientes puntos
        if len(ticket_perf) >= 3:
            z = np.polyfit(x, sellout_values, 2)
            p = np.poly1d(z)
            x_smooth = np.linspace(x.min(), x.max(), 50)
            ax5.plot(x_smooth, p(x_smooth), '--', color=colors['sellout'], alpha=0.5, linewidth=2)
            
            z2 = np.polyfit(x, str_values, 2)
            p2 = np.poly1d(z2)
            ax5.plot(x_smooth, p2(x_smooth), '--', color=colors['str'], alpha=0.5, linewidth=2)
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.95])
        plt.show()
        plt.close(fig)
        
        # Imprimir resumen mejorado
        best_sellout = ticket_perf['tasa_sellout'].idxmax()
        best_str = ticket_perf['sell_through_rate'].idxmax()
        most_common = ticket_perf['num_ofertas'].idxmax()
        best_gmv = ticket_perf['gmv_total'].idxmax()
        
        print()
        print("╔" + "═" * 68 + "╗")
        print("║" + " 💵 PERFORMANCE POR RANGO DE TICKET ".center(68) + "║")
        print("╠" + "═" * 68 + "╣")
        print("║" + f" 🏆 Mayor sellout:      {best_sellout} ({ticket_perf.loc[best_sellout, 'tasa_sellout']:.1%})".ljust(68) + "║")
        print("║" + f" 📈 Mayor sell-through: {best_str} ({ticket_perf.loc[best_str, 'sell_through_rate']:.1%})".ljust(68) + "║")
        print("║" + f" 📊 Rango más común:    {most_common} ({int(ticket_perf.loc[most_common, 'num_ofertas']):,} ofertas)".ljust(68) + "║")
        print("║" + f" 💰 Mayor GMV:          {best_gmv} (${ticket_perf.loc[best_gmv, 'gmv_total']:,.0f})".ljust(68) + "║")
        print("╠" + "═" * 68 + "╣")
        print("║" + f"  {'Rango':<10} {'Ofertas':>9} {'Sellout':>10} {'STR':>10} {'GMV Total':>16}  " + "║")
        print("╠" + "─" * 68 + "╣")
        for idx in ticket_perf.index:
            row = f"  {str(idx):<10} {int(ticket_perf.loc[idx, 'num_ofertas']):>9,} {ticket_perf.loc[idx, 'tasa_sellout']:>9.1%} {ticket_perf.loc[idx, 'sell_through_rate']:>9.1%} ${ticket_perf.loc[idx, 'gmv_total']:>14,.0f}  "
            print("║" + row + "║")
        print("╚" + "═" * 68 + "╝")


# =============================================================================
# TEMÁTICA 6: ESTRATEGIA E IMPACTO (Parte 1 - Envío)
# =============================================================================
# Clases: AnalisisEnvio, AnalisisNegocio, AnalisisDerivado
# Preguntas: Free shipping, riesgo operativo, FOMO, productividad
# =============================================================================

class AnalisisEnvio:
    """Métodos para analizar el impacto del envío gratis."""
    
    @staticmethod
    def get_shipping_performance(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compara performance entre free shipping y sin free shipping.
        """
        shipping = df.groupby('has_free_shipping').agg({
            'OFFER_START_DATE': 'count',
            'has_sales': 'mean',
            'is_sold_out': 'mean',
            'SOLD_AMOUNT': ['sum', 'mean'],
            'sell_through_rate': 'mean',
            'avg_ticket': 'mean'
        })
        shipping.columns = ['total_ofertas', 'tasa_conversion', 'tasa_sellout',
                            'gmv_total', 'gmv_promedio', 'sell_through_rate', 'ticket_promedio']
        shipping.index = ['Sin Free Shipping', 'Con Free Shipping']
        return shipping.round(4)
    
    @staticmethod
    def get_shipping_by_vertical(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compara el efecto del free shipping por vertical.
        """
        results = []
        for vertical in df['VERTICAL'].unique():
            df_v = df[df['VERTICAL'] == vertical]
            conv_fs = df_v[df_v['has_free_shipping']]['has_sales'].mean()
            conv_no_fs = df_v[~df_v['has_free_shipping']]['has_sales'].mean()
            
            if pd.notna(conv_fs) and pd.notna(conv_no_fs):
                results.append({
                    'vertical': vertical,
                    'conv_free_shipping': conv_fs,
                    'conv_no_free_shipping': conv_no_fs,
                    'diferencia_pp': (conv_fs - conv_no_fs) * 100,
                    'fs_mejor': conv_fs > conv_no_fs
                })
        
        return pd.DataFrame(results).sort_values('diferencia_pp', ascending=False)
    
    @staticmethod
    def get_shipping_by_domain(df: pd.DataFrame, min_offers: int = 10) -> pd.DataFrame:
        """
        Compara el efecto del free shipping por dominio.
        """
        # Pivot para comparar FS vs NoFS por dominio
        domain_fs = df.groupby(['DOM_DOMAIN_AGG1', 'has_free_shipping']).agg({
            'has_sales': ['count', 'mean']
        }).reset_index()
        domain_fs.columns = ['domain', 'has_fs', 'count', 'conv_rate']
        
        # Pivot
        pivot = domain_fs.pivot(index='domain', columns='has_fs', values=['count', 'conv_rate'])
        pivot.columns = ['count_no_fs', 'count_fs', 'conv_no_fs', 'conv_fs']
        pivot = pivot.dropna()
        
        # Filtrar dominios con suficientes ofertas
        pivot = pivot[(pivot['count_no_fs'] >= min_offers) & (pivot['count_fs'] >= min_offers)]
        pivot['diff_conv'] = pivot['conv_fs'] - pivot['conv_no_fs']
        pivot['fs_mejor'] = pivot['diff_conv'] > 0
        
        return pivot.sort_values('diff_conv', ascending=False).round(4)
    
    @staticmethod
    def get_low_ticket_fs_analysis(df: pd.DataFrame, percentile: float = 0.25) -> Dict:
        """
        Analiza el subsidio de envíos en tickets bajos.
        """
        df_con_ventas = df[df['has_sales']].copy()
        low_ticket_threshold = df_con_ventas['avg_ticket'].quantile(percentile)
        
        low_ticket_fs = df_con_ventas[(df_con_ventas['avg_ticket'] <= low_ticket_threshold) & 
                                       (df_con_ventas['has_free_shipping'])]
        
        return {
            'umbral_ticket_bajo': low_ticket_threshold,
            'ofertas_low_ticket_fs': len(low_ticket_fs),
            'gmv_low_ticket_fs': low_ticket_fs['SOLD_AMOUNT'].sum(),
            'pct_del_total': len(low_ticket_fs) / len(df_con_ventas) * 100
        }
    
    @staticmethod
    def plot_shipping_comparison(df: pd.DataFrame, figsize: Tuple = (14, 8)) -> None:
        """
        Genera gráfico comparativo de free shipping.
        
        PREGUNTA: ¿El envío gratis mejora la conversión de las ofertas?
        """
        shipping = AnalisisEnvio.get_shipping_performance(df)
        
        # Colores profesionales
        colors = {
            'sin_fs': '#64748B',      # Gris azulado
            'con_fs': '#10B981',      # Verde esmeralda
            'conversion': '#3B82F6',  # Azul
            'sellout': '#F59E0B',     # Naranja/Amber
            'gmv': '#8B5CF6',         # Violeta
            'positive': '#10B981',    # Verde
            'negative': '#EF4444'     # Rojo
        }
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1], hspace=0.35, wspace=0.3)
        
        fig.suptitle('¿El envío gratis mejora la conversión de las ofertas?', 
                    fontsize=16, fontweight='bold', color='#1E293B', y=0.98)
        
        fs = shipping.loc['Con Free Shipping']
        no_fs = shipping.loc['Sin Free Shipping']
        
        # ═══════════════════════════════════════════════════════════════
        # Panel 1: Comparativa de tasas (Conversión y Sellout)
        # ═══════════════════════════════════════════════════════════════
        ax1 = fig.add_subplot(gs[0, :2])
        
        metrics = ['Conversión', 'Sellout']
        sin_fs_vals = [no_fs['tasa_conversion'] * 100, no_fs['tasa_sellout'] * 100]
        con_fs_vals = [fs['tasa_conversion'] * 100, fs['tasa_sellout'] * 100]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, sin_fs_vals, width, label='Sin Free Shipping', 
                       color=colors['sin_fs'], edgecolor='white', linewidth=1.5, zorder=3)
        bars2 = ax1.bar(x + width/2, con_fs_vals, width, label='Con Free Shipping', 
                       color=colors['con_fs'], edgecolor='white', linewidth=1.5, zorder=3)
        
        # Anotaciones en barras
        for bar, val in zip(bars1, sin_fs_vals):
            ax1.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 5), textcoords='offset points', ha='center', va='bottom',
                        fontsize=11, fontweight='bold', color=colors['sin_fs'])
        
        for bar, val in zip(bars2, con_fs_vals):
            ax1.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 5), textcoords='offset points', ha='center', va='bottom',
                        fontsize=11, fontweight='bold', color=colors['con_fs'])
        
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics, fontsize=12, fontweight='medium')
        ax1.set_ylabel('Porcentaje (%)', fontsize=11)
        ax1.set_title('Tasas de Conversión y Sellout', fontsize=13, fontweight='bold', pad=10)
        ax1.legend(loc='upper right', framealpha=0.95, fontsize=10)
        ax1.set_ylim(0, max(sin_fs_vals + con_fs_vals) * 1.25)
        ax1.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
        ax1.set_axisbelow(True)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # ═══════════════════════════════════════════════════════════════
        # Panel 2: Diferencia en puntos porcentuales (bullet chart style)
        # ═══════════════════════════════════════════════════════════════
        ax2 = fig.add_subplot(gs[0, 2])
        
        diff_conv = (fs['tasa_conversion'] - no_fs['tasa_conversion']) * 100
        diff_sellout = (fs['tasa_sellout'] - no_fs['tasa_sellout']) * 100
        
        diffs = [diff_conv, diff_sellout]
        y_pos = [1, 0]
        bar_colors = [colors['positive'] if d > 0 else colors['negative'] for d in diffs]
        
        bars_diff = ax2.barh(y_pos, diffs, height=0.5, color=bar_colors, edgecolor='white', linewidth=1.5, zorder=3)
        ax2.axvline(x=0, color='#94A3B8', linewidth=1.5, linestyle='-', zorder=2)
        
        # Anotaciones de diferencia
        for bar, diff, yp in zip(bars_diff, diffs, y_pos):
            sign = '+' if diff > 0 else ''
            x_pos = diff + (0.3 if diff > 0 else -0.3)
            ha = 'left' if diff > 0 else 'right'
            ax2.annotate(f'{sign}{diff:.1f}pp', xy=(x_pos, yp),
                        ha=ha, va='center', fontsize=11, fontweight='bold',
                        color=colors['positive'] if diff > 0 else colors['negative'])
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(['Conversión', 'Sellout'], fontsize=11)
        ax2.set_xlabel('Diferencia (pp)', fontsize=10)
        ax2.set_title('Impacto del Free Shipping\n(vs Sin FS)', fontsize=12, fontweight='bold', pad=10)
        ax2.grid(axis='x', alpha=0.3, linestyle='--', zorder=0)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # Ajustar límites simétricos
        max_abs = max(abs(min(diffs)), abs(max(diffs))) * 1.5
        ax2.set_xlim(-max_abs, max_abs)
        
        # ═══════════════════════════════════════════════════════════════
        # Panel 3: GMV Total comparativo
        # ═══════════════════════════════════════════════════════════════
        ax3 = fig.add_subplot(gs[1, 0])
        
        gmv_vals = [no_fs['gmv_total'], fs['gmv_total']]
        labels_gmv = ['Sin FS', 'Con FS']
        bar_colors_gmv = [colors['sin_fs'], colors['con_fs']]
        
        bars_gmv = ax3.bar(labels_gmv, gmv_vals, color=bar_colors_gmv, edgecolor='white', linewidth=1.5, zorder=3)
        
        for bar, val in zip(bars_gmv, gmv_vals):
            if val >= 1_000_000:
                label = f'${val/1_000_000:.1f}M'
            else:
                label = f'${val/1_000:,.0f}K'
            ax3.annotate(label, xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 5), textcoords='offset points', ha='center', va='bottom',
                        fontsize=11, fontweight='bold', color='#1E293B')
        
        ax3.set_ylabel('GMV Total ($)', fontsize=10)
        ax3.set_title('GMV Total', fontsize=12, fontweight='bold', pad=10)
        ax3.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1_000_000:.1f}M' if x >= 1_000_000 else f'${x/1_000:.0f}K'))
        
        # ═══════════════════════════════════════════════════════════════
        # Panel 4: GMV Promedio y Ticket
        # ═══════════════════════════════════════════════════════════════
        ax4 = fig.add_subplot(gs[1, 1])
        
        metrics_extra = ['GMV Prom/Oferta', 'Ticket Promedio']
        sin_fs_extra = [no_fs['gmv_promedio'], no_fs['ticket_promedio']]
        con_fs_extra = [fs['gmv_promedio'], fs['ticket_promedio']]
        
        x_extra = np.arange(len(metrics_extra))
        
        bars_e1 = ax4.bar(x_extra - width/2, sin_fs_extra, width, label='Sin FS', 
                         color=colors['sin_fs'], edgecolor='white', linewidth=1.5, zorder=3)
        bars_e2 = ax4.bar(x_extra + width/2, con_fs_extra, width, label='Con FS', 
                         color=colors['con_fs'], edgecolor='white', linewidth=1.5, zorder=3)
        
        for bar, val in zip(bars_e1, sin_fs_extra):
            ax4.annotate(f'${val:,.0f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points', ha='center', va='bottom',
                        fontsize=9, fontweight='bold', color=colors['sin_fs'])
        
        for bar, val in zip(bars_e2, con_fs_extra):
            ax4.annotate(f'${val:,.0f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points', ha='center', va='bottom',
                        fontsize=9, fontweight='bold', color=colors['con_fs'])
        
        ax4.set_xticks(x_extra)
        ax4.set_xticklabels(metrics_extra, fontsize=10)
        ax4.set_ylabel('Valor ($)', fontsize=10)
        ax4.set_title('Métricas de Valor Promedio', fontsize=12, fontweight='bold', pad=10)
        ax4.legend(loc='upper right', fontsize=9)
        ax4.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        
        # ═══════════════════════════════════════════════════════════════
        # Panel 5: Sell-through Rate y volumen de ofertas
        # ═══════════════════════════════════════════════════════════════
        ax5 = fig.add_subplot(gs[1, 2])
        
        # Donut chart para distribución de ofertas
        ofertas = [no_fs['total_ofertas'], fs['total_ofertas']]
        total_ofertas = sum(ofertas)
        
        wedges, texts, autotexts = ax5.pie(
            ofertas, 
            labels=['Sin FS', 'Con FS'],
            colors=[colors['sin_fs'], colors['con_fs']],
            autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*total_ofertas):,})',
            startangle=90,
            wedgeprops=dict(width=0.6, edgecolor='white', linewidth=2),
            textprops={'fontsize': 10},
            pctdistance=0.75
        )
        
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_fontweight('bold')
        
        ax5.set_title(f'Distribución de Ofertas\n(Total: {int(total_ofertas):,})', 
                     fontsize=12, fontweight='bold', pad=10)
        
        # ═══════════════════════════════════════════════════════════════
        # Conclusión visual al pie
        # ═══════════════════════════════════════════════════════════════
        if diff_conv > 0:
            conclusion_emoji = "✅"
            conclusion_color = colors['positive']
            conclusion_text = f"Free Shipping MEJORA la conversión en {diff_conv:.1f}pp"
        else:
            conclusion_emoji = "⚠️"
            conclusion_color = colors['negative']
            conclusion_text = f"Free Shipping REDUCE la conversión en {abs(diff_conv):.1f}pp"
        
        fig.text(0.5, 0.02, f"{conclusion_emoji} {conclusion_text}", 
                ha='center', va='center', fontsize=13, fontweight='bold',
                color=conclusion_color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8FAFC', edgecolor=conclusion_color, linewidth=2))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.show()
        plt.close(fig)
        
        # Imprimir resumen
        print()
        print("=" * 70)
        print("🚚 IMPACTO DEL FREE SHIPPING")
        print("=" * 70)
        print(f"{'Métrica':<25} {'Sin FS':>15} {'Con FS':>15} {'Diferencia':>12}")
        print("-" * 70)
        print(f"{'Ofertas':<25} {int(no_fs['total_ofertas']):>15,} {int(fs['total_ofertas']):>15,}")
        print(f"{'Tasa Conversión':<25} {no_fs['tasa_conversion']:>14.1%} {fs['tasa_conversion']:>14.1%} {diff_conv:>+11.1f}pp")
        print(f"{'Tasa Sellout':<25} {no_fs['tasa_sellout']:>14.1%} {fs['tasa_sellout']:>14.1%} {diff_sellout:>+11.1f}pp")
        print(f"{'GMV Total':<25} ${no_fs['gmv_total']:>14,.0f} ${fs['gmv_total']:>13,.0f}")
        print(f"{'GMV Promedio':<25} ${no_fs['gmv_promedio']:>14,.0f} ${fs['gmv_promedio']:>13,.0f}")
        print(f"{'Ticket Promedio':<25} ${no_fs['ticket_promedio']:>14,.0f} ${fs['ticket_promedio']:>13,.0f}")
        print(f"{'Sell-through Rate':<25} {no_fs['sell_through_rate']:>14.1%} {fs['sell_through_rate']:>14.1%}")
        print("-" * 70)
        if diff_conv > 0:
            print(f"✅ Free Shipping MEJORA la conversión en {diff_conv:.1f} puntos porcentuales")
        else:
            print(f"⚠️ Free Shipping REDUCE la conversión en {abs(diff_conv):.1f} puntos porcentuales")
        print("=" * 70)
    
    @staticmethod
    def plot_shipping_by_vertical(df: pd.DataFrame, figsize: Tuple = (16, 10)) -> None:
        """
        Genera gráfico de efecto del free shipping por vertical.
        
        PREGUNTA: ¿En qué verticales el envío gratis tiene mayor impacto?
        """
        shipping_v = AnalisisEnvio.get_shipping_by_vertical(df)
        
        # Colores profesionales
        colors = {
            'sin_fs': '#64748B',      # Gris azulado
            'con_fs': '#10B981',      # Verde esmeralda
            'positive': '#10B981',    # Verde
            'negative': '#EF4444',    # Rojo
            'neutral': '#94A3B8'      # Gris neutro
        }
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.5, 0.8], wspace=0.25)
        
        fig.suptitle('¿En qué verticales el envío gratis tiene mayor impacto?', 
                    fontsize=16, fontweight='bold', color='#1E293B', y=0.98)
        
        n_verticals = len(shipping_v)
        
        # ═══════════════════════════════════════════════════════════════
        # Panel 1: Tasas de conversión lado a lado
        # ═══════════════════════════════════════════════════════════════
        ax1 = fig.add_subplot(gs[0, 0])
        
        y_pos = np.arange(n_verticals)
        height = 0.35
        
        bars_sin = ax1.barh(y_pos + height/2, shipping_v['conv_no_free_shipping'] * 100, height,
                           label='Sin Free Shipping', color=colors['sin_fs'], 
                           edgecolor='white', linewidth=1, zorder=3)
        bars_con = ax1.barh(y_pos - height/2, shipping_v['conv_free_shipping'] * 100, height,
                           label='Con Free Shipping', color=colors['con_fs'], 
                           edgecolor='white', linewidth=1, zorder=3)
        
        # Anotaciones de valores
        for bar, val in zip(bars_sin, shipping_v['conv_no_free_shipping'] * 100):
            ax1.annotate(f'{val:.1f}%', xy=(val, bar.get_y() + bar.get_height()/2),
                        xytext=(3, 0), textcoords='offset points', ha='left', va='center',
                        fontsize=8, color=colors['sin_fs'], fontweight='medium')
        
        for bar, val in zip(bars_con, shipping_v['conv_free_shipping'] * 100):
            ax1.annotate(f'{val:.1f}%', xy=(val, bar.get_y() + bar.get_height()/2),
                        xytext=(3, 0), textcoords='offset points', ha='left', va='center',
                        fontsize=8, color=colors['con_fs'], fontweight='medium')
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(shipping_v['vertical'], fontsize=9)
        ax1.set_xlabel('Tasa de Conversión (%)', fontsize=10)
        ax1.set_title('Conversión por Tipo de Envío', fontsize=12, fontweight='bold', pad=10)
        ax1.legend(loc='lower right', fontsize=9, framealpha=0.95)
        ax1.grid(axis='x', alpha=0.3, linestyle='--', zorder=0)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.invert_yaxis()
        
        # ═══════════════════════════════════════════════════════════════
        # Panel 2: Diferencia en puntos porcentuales (diverging bar)
        # ═══════════════════════════════════════════════════════════════
        ax2 = fig.add_subplot(gs[0, 1])
        
        bar_colors = [colors['positive'] if x > 0 else colors['negative'] for x in shipping_v['diferencia_pp']]
        
        bars_diff = ax2.barh(y_pos, shipping_v['diferencia_pp'], height=0.6, 
                            color=bar_colors, edgecolor='white', linewidth=1.5, zorder=3)
        
        # Línea central en cero
        ax2.axvline(x=0, color='#475569', linewidth=2, linestyle='-', zorder=2)
        
        # Anotaciones de diferencia
        for bar, diff, yp in zip(bars_diff, shipping_v['diferencia_pp'], y_pos):
            sign = '+' if diff > 0 else ''
            x_offset = 0.5 if diff > 0 else -0.5
            ha = 'left' if diff > 0 else 'right'
            color = colors['positive'] if diff > 0 else colors['negative']
            ax2.annotate(f'{sign}{diff:.1f}pp', xy=(diff + x_offset, yp),
                        ha=ha, va='center', fontsize=9, fontweight='bold', color=color)
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(shipping_v['vertical'], fontsize=9)
        ax2.set_xlabel('Diferencia en Conversión (pp)', fontsize=10)
        ax2.set_title('Impacto del Free Shipping\n(Positivo = FS mejora conversión)', fontsize=12, fontweight='bold', pad=10)
        ax2.grid(axis='x', alpha=0.3, linestyle='--', zorder=0)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.invert_yaxis()
        
        # Ajustar límites simétricos
        max_abs = max(abs(shipping_v['diferencia_pp'].min()), abs(shipping_v['diferencia_pp'].max())) * 1.3
        ax2.set_xlim(-max_abs, max_abs)
        
        # Zonas de fondo coloreadas
        ax2.axvspan(0, max_abs, alpha=0.08, color=colors['positive'], zorder=0)
        ax2.axvspan(-max_abs, 0, alpha=0.08, color=colors['negative'], zorder=0)
        
        # ═══════════════════════════════════════════════════════════════
        # Panel 3: Resumen estadístico
        # ═══════════════════════════════════════════════════════════════
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        
        fs_helps = (shipping_v['diferencia_pp'] > 0).sum()
        fs_hurts = (shipping_v['diferencia_pp'] < 0).sum()
        avg_impact = shipping_v['diferencia_pp'].mean()
        max_positive = shipping_v['diferencia_pp'].max()
        max_negative = shipping_v['diferencia_pp'].min()
        best_vertical = shipping_v.iloc[0]['vertical']
        worst_vertical = shipping_v.iloc[-1]['vertical']
        
        # Crear texto de resumen con estilo de tarjetas
        summary_text = f"""
RESUMEN DEL IMPACTO

━━━━━━━━━━━━━━━━━━━━━━━

🟢 FS Ayuda
   {fs_helps} verticales

🔴 FS Perjudica
   {fs_hurts} verticales

━━━━━━━━━━━━━━━━━━━━━━━

📊 Impacto Promedio
   {avg_impact:+.1f}pp

📈 Mayor Beneficio
   {max_positive:+.1f}pp
   {best_vertical[:18]}

📉 Mayor Perjuicio
   {max_negative:+.1f}pp
   {worst_vertical[:18]}

━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='#F8FAFC', 
                         edgecolor='#E2E8F0', linewidth=2))
        
        # ═══════════════════════════════════════════════════════════════
        # Conclusión visual al pie
        # ═══════════════════════════════════════════════════════════════
        if fs_helps > fs_hurts:
            conclusion_emoji = "✅"
            conclusion_color = colors['positive']
            conclusion_text = f"Free Shipping beneficia a {fs_helps}/{n_verticals} verticales (impacto promedio: {avg_impact:+.1f}pp)"
        elif fs_hurts > fs_helps:
            conclusion_emoji = "⚠️"
            conclusion_color = colors['negative']
            conclusion_text = f"Free Shipping perjudica a {fs_hurts}/{n_verticals} verticales (impacto promedio: {avg_impact:+.1f}pp)"
        else:
            conclusion_emoji = "➖"
            conclusion_color = colors['neutral']
            conclusion_text = f"Free Shipping tiene impacto mixto: {fs_helps} beneficiadas, {fs_hurts} perjudicadas"
        
        fig.text(0.5, 0.02, f"{conclusion_emoji} {conclusion_text}", 
                ha='center', va='center', fontsize=12, fontweight='bold',
                color=conclusion_color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8FAFC', 
                         edgecolor=conclusion_color, linewidth=2))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.show()
        plt.close(fig)
        
        # Imprimir resumen
        print()
        print("=" * 70)
        print("🚚 EFECTO FREE SHIPPING POR VERTICAL")
        print("=" * 70)
        print(f"Verticales donde FS ayuda:      {fs_helps}")
        print(f"Verticales donde FS perjudica:  {fs_hurts}")
        print(f"Impacto promedio:               {avg_impact:+.1f}pp")
        print("-" * 70)
        print(f"{'Vertical':<25} {'Sin FS':>12} {'Con FS':>12} {'Diff (pp)':>12}")
        print("-" * 70)
        for _, row in shipping_v.iterrows():
            emoji = "🟢" if row['diferencia_pp'] > 0 else "🔴"
            print(f"{row['vertical'][:25]:<25} {row['conv_no_free_shipping']:>11.1%} {row['conv_free_shipping']:>11.1%} {emoji}{row['diferencia_pp']:>+10.1f}")
        print("-" * 70)
        print(f"{'PROMEDIO':<25} {shipping_v['conv_no_free_shipping'].mean():>11.1%} {shipping_v['conv_free_shipping'].mean():>11.1%} {avg_impact:>+11.1f}")
        print("=" * 70)


# =============================================================================
# TEMÁTICA 5: STOCK Y OPERACIONES (Parte 1)
# =============================================================================
# Clases: AnalisisStock, AnalisisOrigen
# Preguntas: Stock óptimo, sobreventas, campo ORIGIN
# =============================================================================

class AnalisisStock:
    """Métodos para análisis de stock óptimo y sobreventas."""
    
    @staticmethod
    def get_stock_performance(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula la performance por rangos de stock.
        """
        stock_perf = df.groupby('stock_bucket').agg({
            'OFFER_START_DATE': 'count',
            'has_sales': 'mean',
            'is_sold_out': 'mean',
            'sell_through_rate': 'mean',
            'has_oversell': 'mean',
            'SOLD_AMOUNT': 'sum',
            'gmv_per_committed_unit': 'mean'
        })
        stock_perf.columns = ['total_ofertas', 'tasa_conversion', 'tasa_sellout',
                              'sell_through', 'tasa_oversell', 'gmv_total', 'gmv_por_unidad']
        return stock_perf.round(4)
    
    @staticmethod
    def get_oversell_analysis(df: pd.DataFrame) -> Dict:
        """
        Análisis detallado de sobreventas.
        """
        oversell_df = df[df['has_oversell']]
        
        return {
            'total_oversell': len(oversell_df),
            'pct_oversell': len(oversell_df) / len(df) * 100,
            'unidades_oversell': oversell_df['oversell_qty'].sum(),
            'gmv_oversell': oversell_df['SOLD_AMOUNT'].sum(),
            'avg_ticket_oversell': oversell_df['avg_ticket'].mean(),
            'top_categories': oversell_df.groupby('DOM_DOMAIN_AGG1')['oversell_qty'].sum().sort_values(ascending=False).head(10)
        }
    
    @staticmethod
    def get_stock_efficiency(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula la eficiencia del stock (GMV / Stock comprometido).
        """
        df_con_ventas = df[df['has_sales']].copy()
        
        efficiency = df_con_ventas.groupby('stock_bucket').agg({
            'SOLD_AMOUNT': 'sum',
            'INVOLVED_STOCK': 'sum',
            'SOLD_QUANTITY': 'sum'
        })
        efficiency['gmv_per_unit_committed'] = efficiency['SOLD_AMOUNT'] / efficiency['INVOLVED_STOCK']
        efficiency['gmv_per_unit_sold'] = efficiency['SOLD_AMOUNT'] / efficiency['SOLD_QUANTITY']
        efficiency['sell_through'] = efficiency['SOLD_QUANTITY'] / efficiency['INVOLVED_STOCK']
        
        return efficiency.round(4)
    
    @staticmethod
    def plot_stock_performance(df: pd.DataFrame, figsize: Tuple = (16, 5)) -> None:
        """
        Genera gráfico de performance por nivel de stock con anotaciones.
        
        PREGUNTA: ¿Cuál es el nivel óptimo de stock para maximizar resultados?
        """
        stock_perf = AnalisisStock.get_stock_performance(df)
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle('¿Cuál es el nivel óptimo de stock para maximizar resultados?', 
                    fontsize=14, fontweight='bold', color='#1A252F', y=1.02)
        
        x_labels = stock_perf.index.astype(str)
        
        # Identificar máximos para resaltarlos
        best_conv_idx = stock_perf['tasa_conversion'].idxmax()
        best_sellout_idx = stock_perf['tasa_sellout'].idxmax()
        best_eff_idx = stock_perf['gmv_por_unidad'].idxmax()
        
        # Gráfico 1: Tasa de conversión
        colors1 = [COLORS['success'] if idx == best_conv_idx else COLORS['primary'] 
                   for idx in stock_perf.index]
        bars1 = axes[0].bar(x_labels, stock_perf['tasa_conversion'] * 100, 
                           color=colors1, edgecolor='white', linewidth=1)
        axes[0].set_xlabel('Stock Comprometido', fontweight='medium')
        axes[0].set_ylabel('Tasa de Conversión (%)', fontweight='medium')
        axes[0].set_title('Conversión por Nivel de Stock', pad=10)
        axes[0].tick_params(axis='x', rotation=45)
        # Agregar valores
        for bar, val in zip(bars1, stock_perf['tasa_conversion'] * 100):
            axes[0].text(bar.get_x() + bar.get_width()/2, val + 1, 
                        f'{val:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Gráfico 2: Tasa de sellout
        colors2 = [COLORS['success'] if idx == best_sellout_idx else COLORS['secondary'] 
                   for idx in stock_perf.index]
        bars2 = axes[1].bar(x_labels, stock_perf['tasa_sellout'] * 100, 
                           color=colors2, edgecolor='white', linewidth=1)
        axes[1].set_xlabel('Stock Comprometido', fontweight='medium')
        axes[1].set_ylabel('Tasa de Sellout (%)', fontweight='medium')
        axes[1].set_title('Sellout por Nivel de Stock', pad=10)
        axes[1].tick_params(axis='x', rotation=45)
        # Agregar valores
        for bar, val in zip(bars2, stock_perf['tasa_sellout'] * 100):
            axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.5, 
                        f'{val:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Gráfico 3: GMV por unidad
        colors3 = [COLORS['success'] if idx == best_eff_idx else COLORS['accent1'] 
                   for idx in stock_perf.index]
        bars3 = axes[2].bar(x_labels, stock_perf['gmv_por_unidad'], 
                           color=colors3, edgecolor='white', linewidth=1)
        axes[2].set_xlabel('Stock Comprometido', fontweight='medium')
        axes[2].set_ylabel('GMV por Unidad ($)', fontweight='medium')
        axes[2].set_title('GMV por Nivel de Stock', pad=10)
        axes[2].tick_params(axis='x', rotation=45)
        # Agregar valores
        for bar, val in zip(bars3, stock_perf['gmv_por_unidad']):
            axes[2].text(bar.get_x() + bar.get_width()/2, val + 0.3, 
                        f'${val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Agregar leyenda explicativa
        fig.text(0.5, -0.02, '★ Barras verdes indican el mejor rendimiento en cada métrica', 
                ha='center', fontsize=10, style='italic', color=COLORS['dark'], alpha=0.7)
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        
        # Imprimir resumen
        best_conv = stock_perf['tasa_conversion'].idxmax()
        best_sellout = stock_perf['tasa_sellout'].idxmax()
        best_efficiency = stock_perf['gmv_por_unidad'].idxmax()
        print()
        print("=" * 60)
        print("📦 PERFORMANCE POR NIVEL DE STOCK")
        print("=" * 60)
        print(f"🏆 Mejor conversión:   Stock {best_conv} ({stock_perf.loc[best_conv, 'tasa_conversion']:.1%})")
        print(f"🎯 Mayor sellout:      Stock {best_sellout} ({stock_perf.loc[best_sellout, 'tasa_sellout']:.1%})")
        print(f"💰 Mayor GMV/unidad:   Stock {best_efficiency} (${stock_perf.loc[best_efficiency, 'gmv_por_unidad']:.2f}/unidad)")
        print("-" * 60)
        print("📋 Resumen por nivel:")
        print(f"{'Stock':<12} {'Ofertas':>10} {'Conversión':>12} {'Sellout':>10} {'GMV/Unidad':>12}")
        print("-" * 60)
        for idx in stock_perf.index:
            print(f"{str(idx):<12} {int(stock_perf.loc[idx, 'total_ofertas']):>10,} {stock_perf.loc[idx, 'tasa_conversion']:>11.1%} {stock_perf.loc[idx, 'tasa_sellout']:>9.1%} ${stock_perf.loc[idx, 'gmv_por_unidad']:>10.2f}")
        print("=" * 60)
    
    @staticmethod
    def plot_oversell_by_category(df: pd.DataFrame, figsize: Tuple = (12, 6)) -> None:
        """
        Genera gráfico de sobreventas por categoría.
        
        PREGUNTA: ¿Qué categorías tienen más problemas de sobreventa?
        """
        oversell = AnalisisStock.get_oversell_analysis(df)
        top_cats = oversell['top_categories']
        
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle('¿Qué categorías tienen más problemas de sobreventa?', 
                    fontsize=14, fontweight='bold', color='#1A252F', y=1.02)
        
        ax.barh(top_cats.index, top_cats.values, color='#e74c3c')
        ax.set_xlabel('Unidades en Sobreventa')
        ax.set_title('Top Categorías con Sobreventas')
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        
        # Imprimir resumen
        print()
        print("=" * 60)
        print("⚠️ ANÁLISIS DE SOBREVENTAS (OVERSELL)")
        print("=" * 60)
        print(f"Total ofertas con oversell:  {oversell['total_oversell']:,} ({oversell['pct_oversell']:.1f}%)")
        print(f"Unidades sobrevendidas:      {int(oversell['unidades_oversell']):,}")
        print(f"GMV en riesgo:               ${oversell['gmv_oversell']:,.0f}")
        print(f"Ticket promedio oversell:    ${oversell['avg_ticket_oversell']:.2f}")
        print("-" * 60)
        print("📋 Top 10 categorías con más unidades sobrevendidas:")
        for i, (idx, val) in enumerate(top_cats.items(), 1):
            print(f"   {i}. {idx[:40]}: {int(val):,} unidades")
        print("=" * 60)


# =============================================================================
# TEMÁTICA 5: STOCK Y OPERACIONES (Parte 2 - Origen)
# =============================================================================

class AnalisisOrigen:
    """Métodos para analizar el campo ORIGIN."""
    
    @staticmethod
    def get_origin_distribution(df: pd.DataFrame) -> Dict:
        """
        Analiza la distribución del campo ORIGIN.
        """
        origin_counts = df['ORIGIN'].value_counts(dropna=False)
        
        return {
            'distribution': origin_counts,
            'null_count': df['ORIGIN'].isna().sum(),
            'null_pct': df['ORIGIN'].isna().mean() * 100,
            'unique_values': df['ORIGIN'].dropna().unique().tolist()
        }
    
    @staticmethod
    def get_origin_performance(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compara performance entre diferentes orígenes.
        """
        # Crear columna para nulos
        df_copy = df.copy()
        df_copy['ORIGIN_CLEAN'] = df_copy['ORIGIN'].fillna('NULL')
        
        origin_perf = df_copy.groupby('ORIGIN_CLEAN').agg({
            'OFFER_START_DATE': 'count',
            'has_sales': 'mean',
            'is_sold_out': 'mean',
            'has_oversell': 'mean',
            'SOLD_AMOUNT': ['sum', 'mean'],
            'sell_through_rate': 'mean'
        })
        origin_perf.columns = ['total_ofertas', 'tasa_conversion', 'tasa_sellout',
                               'tasa_oversell', 'gmv_total', 'gmv_promedio', 'sell_through_rate']
        return origin_perf.round(4)
    
    @staticmethod
    def plot_origin_analysis(df: pd.DataFrame, figsize: Tuple = (12, 5)) -> None:
        """
        Genera gráfico de análisis de origen.
        
        PREGUNTA: ¿Cómo impacta el campo ORIGIN en la performance?
        """
        origin_perf = AnalisisOrigen.get_origin_performance(df)
        origin_dist = AnalisisOrigen.get_origin_distribution(df)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('¿Cómo impacta el campo ORIGIN en la performance?', 
                    fontsize=14, fontweight='bold', color='#1A252F', y=1.02)
        
        # Gráfico 1: Distribución
        axes[0].bar(origin_perf.index, origin_perf['total_ofertas'], color='#3498db')
        axes[0].set_xlabel('Origen')
        axes[0].set_ylabel('Número de Ofertas')
        axes[0].set_title('Distribución por Origen')
        
        # Gráfico 2: Performance
        x = range(len(origin_perf))
        width = 0.25
        axes[1].bar([i - width for i in x], origin_perf['tasa_conversion'] * 100,
                    width, label='Conversión', color='#3498db')
        axes[1].bar(x, origin_perf['tasa_sellout'] * 100,
                    width, label='Sellout', color='#2ecc71')
        axes[1].bar([i + width for i in x], origin_perf['tasa_oversell'] * 100,
                    width, label='Oversell', color='#e74c3c')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(origin_perf.index)
        axes[1].set_ylabel('Porcentaje (%)')
        axes[1].set_title('Performance por Origen')
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        
        # Imprimir resumen
        best_origin = origin_perf['tasa_conversion'].idxmax()
        print()
        print("=" * 60)
        print("🏷️ ANÁLISIS POR CAMPO ORIGIN")
        print("=" * 60)
        print(f"Valores únicos: {origin_dist['unique_values']}")
        print(f"Valores nulos:  {origin_dist['null_count']:,} ({origin_dist['null_pct']:.1f}%)")
        print("-" * 60)
        print(f"{'Origen':<10} {'Ofertas':>12} {'Conversión':>12} {'Sellout':>10} {'GMV Total':>15}")
        print("-" * 60)
        for idx, row in origin_perf.iterrows():
            print(f"{str(idx):<10} {int(row['total_ofertas']):>12,} {row['tasa_conversion']:>11.1%} {row['tasa_sellout']:>9.1%} ${row['gmv_total']:>13,.0f}")
        print("-" * 60)
        print(f"🏆 Mejor origen: {best_origin} ({origin_perf.loc[best_origin, 'tasa_conversion']:.1%} conversión)")
        print("=" * 60)


# =============================================================================
# TEMÁTICA 1: PERFORMANCE Y RESULTADOS (Parte 2 - Zombies)
# =============================================================================

class AnalisisZombies:
    """Métodos para analizar ofertas fallidas (sin ventas)."""
    
    @staticmethod
    def get_zombie_summary(df: pd.DataFrame) -> Dict:
        """
        Resumen de ofertas zombie.
        """
        zombies = df[~df['has_sales']]
        
        return {
            'total_zombies': len(zombies),
            'pct_zombies': len(zombies) / len(df) * 100,
            'stock_desperdiciado': zombies['INVOLVED_STOCK'].sum(),
            'horas_slot_perdidas': zombies['duration_hours'].sum()
        }
    
    @staticmethod
    def get_zombie_by_category(df: pd.DataFrame) -> pd.DataFrame:
        """
        Analiza zombies por categoría.
        """
        zombie_by_cat = df.groupby('DOM_DOMAIN_AGG1').agg({
            'has_sales': lambda x: (~x).sum(),  # Contar zombies
            'OFFER_START_DATE': 'count',
            'INVOLVED_STOCK': lambda x: x[~df.loc[x.index, 'has_sales']].sum()
        })
        zombie_by_cat.columns = ['num_zombies', 'total_ofertas', 'stock_desperdiciado']
        zombie_by_cat['tasa_zombie'] = zombie_by_cat['num_zombies'] / zombie_by_cat['total_ofertas']
        return zombie_by_cat.sort_values('num_zombies', ascending=False).round(4)
    
    @staticmethod
    def get_zombie_temporal_pattern(df: pd.DataFrame) -> pd.DataFrame:
        """
        Analiza patrones temporales de zombies.
        """
        zombie_temporal = df.groupby('start_hour').agg({
            'has_sales': lambda x: (~x).sum(),
            'OFFER_START_DATE': 'count'
        })
        zombie_temporal.columns = ['num_zombies', 'total_ofertas']
        zombie_temporal['tasa_zombie'] = zombie_temporal['num_zombies'] / zombie_temporal['total_ofertas']
        return zombie_temporal.round(4)
    
    @staticmethod
    def estimate_lost_gmv(df: pd.DataFrame, verbose: bool = True) -> Dict:
        """
        Estima el GMV potencial perdido en ofertas zombie.
        """
        zombies = df[~df['has_sales']]
        exitosas = df[df['has_sales']]
        
        # Calcular ticket promedio por categoría
        ticket_by_cat = exitosas.groupby('DOM_DOMAIN_AGG1')['avg_ticket'].mean()
        
        # Estimar GMV perdido
        zombies_copy = zombies.copy()
        zombies_copy['estimated_ticket'] = zombies_copy['DOM_DOMAIN_AGG1'].map(ticket_by_cat)
        zombies_copy['estimated_lost_gmv'] = zombies_copy['INVOLVED_STOCK'] * zombies_copy['estimated_ticket']
        
        result = {
            'gmv_potencial_perdido': float(zombies_copy['estimated_lost_gmv'].sum()),
            'gmv_promedio_exitosas': float(exitosas['SOLD_AMOUNT'].mean()),
            'gmv_potencial_alternativo': float(len(zombies) * exitosas['SOLD_AMOUNT'].mean())
        }
        
        if verbose:
            print("=" * 60)
            print("💀 GMV POTENCIAL PERDIDO EN ZOMBIES")
            print("=" * 60)
            print(f"📊 Ofertas zombie analizadas:     {len(zombies):,}")
            print(f"✅ Ofertas exitosas (referencia): {len(exitosas):,}")
            print("-" * 60)
            print(f"💰 GMV potencial perdido:         ${result['gmv_potencial_perdido']:,.2f}")
            print(f"📈 GMV promedio en exitosas:      ${result['gmv_promedio_exitosas']:,.2f}")
            print(f"🔄 GMV potencial alternativo:     ${result['gmv_potencial_alternativo']:,.2f}")
            print("=" * 60)
        
        return result
    
    @staticmethod
    def plot_zombie_by_category(df: pd.DataFrame, top_n: int = 15, 
                                 figsize: Tuple = (12, 6)) -> None:
        """
        Genera gráfico de zombies por categoría.
        
        PREGUNTA: ¿Qué categorías tienen más ofertas fallidas (zombies)?
        """
        zombie_cat = AnalisisZombies.get_zombie_by_category(df).head(top_n)
        zombie_summary = AnalisisZombies.get_zombie_summary(df)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('¿Qué categorías tienen más ofertas fallidas (zombies)?', 
                    fontsize=14, fontweight='bold', color='#1A252F', y=1.02)
        
        # Gráfico 1: Número de zombies
        axes[0].barh(zombie_cat.index, zombie_cat['num_zombies'], color='#e74c3c')
        axes[0].set_xlabel('Número de Zombies')
        axes[0].set_title(f'Top {top_n} Categorías con más Zombies')
        
        # Gráfico 2: Tasa de zombies
        axes[1].barh(zombie_cat.index, zombie_cat['tasa_zombie'] * 100, color='#e67e22')
        axes[1].set_xlabel('Tasa de Zombies (%)')
        axes[1].set_title('Tasa de Zombies por Categoría')
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        
        # Imprimir resumen
        print()
        print("=" * 60)
        print("💀 ANÁLISIS DE OFERTAS ZOMBIES (SIN VENTAS)")
        print("=" * 60)
        print(f"Total zombies:         {zombie_summary['total_zombies']:,} ({zombie_summary['pct_zombies']:.1f}%)")
        print(f"Stock desperdiciado:   {zombie_summary['stock_desperdiciado']:,} unidades")
        print(f"Horas de slot perdidas: {zombie_summary['horas_slot_perdidas']:,.0f} hrs")
        print("-" * 60)
        print(f"📋 Top {top_n} categorías con más zombies:")
        for i, (idx, row) in enumerate(zombie_cat.head(10).iterrows(), 1):
            print(f"   {i}. {idx[:35]}: {int(row['num_zombies']):,} ({row['tasa_zombie']:.1%})")
        if top_n > 10:
            print(f"   ... y {top_n - 10} más")
        print("=" * 60)


# =============================================================================
# TEMÁTICA 4: PRICING, GMV Y VELOCIDAD (Parte 2 - Velocidad)
# =============================================================================

class AnalisisVelocidad:
    """Métodos para analizar velocidad de venta."""
    
    @staticmethod
    def get_velocity_stats(df: pd.DataFrame) -> Dict:
        """
        Estadísticas de velocidad de venta.
        """
        df_con_ventas = df[(df['has_sales']) & (df['duration_hours'] > 0)]
        
        return {
            'gmv_per_hour_mean': df_con_ventas['gmv_per_hour'].mean(),
            'gmv_per_hour_median': df_con_ventas['gmv_per_hour'].median(),
            'gmv_per_hour_p90': df_con_ventas['gmv_per_hour'].quantile(0.9),
            'gmv_per_hour_p10': df_con_ventas['gmv_per_hour'].quantile(0.1),
            'units_per_hour_mean': df_con_ventas['units_per_hour'].mean(),
            'units_per_hour_median': df_con_ventas['units_per_hour'].median()
        }
    
    @staticmethod
    def get_top_performers(df: pd.DataFrame, top_pct: float = 0.1) -> pd.DataFrame:
        """
        Identifica las ofertas de mayor velocidad.
        """
        df_con_ventas = df[(df['has_sales']) & (df['duration_hours'] > 0)].copy()
        threshold = df_con_ventas['gmv_per_hour'].quantile(1 - top_pct)
        
        top = df_con_ventas[df_con_ventas['gmv_per_hour'] >= threshold]
        return top[['DOMAIN_ID', 'DOM_DOMAIN_AGG1', 'SOLD_AMOUNT', 'duration_hours', 
                    'gmv_per_hour', 'sell_through_rate']].sort_values('gmv_per_hour', ascending=False)
    
    @staticmethod
    def get_bottom_performers(df: pd.DataFrame, bottom_pct: float = 0.1) -> pd.DataFrame:
        """
        Identifica las ofertas de menor velocidad.
        """
        df_con_ventas = df[(df['has_sales']) & (df['duration_hours'] > 0)].copy()
        threshold = df_con_ventas['gmv_per_hour'].quantile(bottom_pct)
        
        bottom = df_con_ventas[df_con_ventas['gmv_per_hour'] <= threshold]
        return bottom[['DOMAIN_ID', 'DOM_DOMAIN_AGG1', 'SOLD_AMOUNT', 'duration_hours',
                       'gmv_per_hour', 'sell_through_rate']].sort_values('gmv_per_hour')
    
    @staticmethod
    def get_time_to_sellout(df: pd.DataFrame) -> pd.DataFrame:
        """
        Analiza el tiempo hasta agotar stock.
        """
        sellout_df = df[df['is_sold_out']].copy()
        
        return sellout_df[['DOMAIN_ID', 'DOM_DOMAIN_AGG1', 'duration_hours', 
                           'SOLD_AMOUNT', 'SOLD_QUANTITY', 'gmv_per_hour']].describe()
    
    @staticmethod
    def plot_velocity_distribution(df: pd.DataFrame, figsize: Tuple = (8, 6)) -> None:
        """
        Genera histograma de velocidad de venta con KDE.
        
        PREGUNTA: ¿Cuál es la velocidad típica de venta (GMV/hora)?
        """
        df_con_ventas = df[(df['has_sales']) & (df['duration_hours'] > 0)].copy()
        velocity_stats = AnalisisVelocidad.get_velocity_stats(df)
        
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle('¿Cuál es la velocidad típica de venta (GMV/hora)?', 
                    fontsize=14, fontweight='bold', color='#1A252F', y=1.02)
        
        # Distribución de GMV por hora con KDE
        gmv_per_hour = df_con_ventas['gmv_per_hour'].dropna()
        gmv_per_hour = gmv_per_hour[gmv_per_hour <= gmv_per_hour.quantile(0.95)]  # Remover outliers
        
        # Histograma con KDE
        n, bins, patches = ax.hist(gmv_per_hour, bins=50, edgecolor='white', 
                                   alpha=0.7, color=COLORS['primary'], density=True)
        
        # KDE overlay
        kde_x = np.linspace(gmv_per_hour.min(), gmv_per_hour.max(), 200)
        kde = stats.gaussian_kde(gmv_per_hour)
        ax.plot(kde_x, kde(kde_x), color=COLORS['accent1'], linewidth=2.5, label='Densidad (KDE)')
        
        # Líneas de referencia
        median_val = gmv_per_hour.median()
        p90_val = gmv_per_hour.quantile(0.9)
        ax.axvline(x=median_val, color=COLORS['secondary'], linestyle='-', linewidth=2,
                   label=f'Mediana: ${median_val:.0f}/hr')
        ax.axvline(x=p90_val, color=COLORS['success'], linestyle='--', linewidth=2, alpha=0.7,
                   label=f'P90: ${p90_val:.0f}/hr')
        
        # Sombrear zona de top performers
        ax.axvspan(p90_val, gmv_per_hour.max(), alpha=0.1, color=COLORS['success'], label='Top 10%')
        
        ax.set_xlabel('GMV por Hora ($)', fontweight='medium')
        ax.set_ylabel('Densidad', fontweight='medium')
        ax.set_title('Distribución de Velocidad de Venta (GMV/hora)', pad=15)
        ax.legend(loc='upper right', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        
        # Imprimir resumen
        print()
        print("=" * 60)
        print("⚡ VELOCIDAD DE VENTA (GMV/HORA)")
        print("=" * 60)
        print(f"Ofertas analizadas: {len(df_con_ventas):,}")
        print("-" * 60)
        print(f"💰 GMV/hora promedio:  ${velocity_stats['gmv_per_hour_mean']:.2f}")
        print(f"💰 GMV/hora mediana:   ${velocity_stats['gmv_per_hour_median']:.2f}")
        print(f"📈 GMV/hora P90:       ${velocity_stats['gmv_per_hour_p90']:.2f}")
        print(f"📉 GMV/hora P10:       ${velocity_stats['gmv_per_hour_p10']:.2f}")
        print("-" * 60)
        print(f"📦 Unidades/hora promedio: {velocity_stats['units_per_hour_mean']:.2f}")
        print(f"📦 Unidades/hora mediana:  {velocity_stats['units_per_hour_median']:.2f}")
        print("=" * 60)


# =============================================================================
# TEMÁTICA 3: CATEGORÍAS Y DOMINIOS (Parte 2 - Canibalización)
# =============================================================================

class AnalisisCanibalizacion:
    """Métodos para analizar canibalización entre ofertas."""
    
    @staticmethod
    def get_concurrent_offers(df: pd.DataFrame) -> pd.DataFrame:
        """
        Cuenta ofertas concurrentes por día y dominio.
        """
        concurrent = df.groupby(['OFFER_START_DATE', 'DOM_DOMAIN_AGG1']).agg({
            'has_sales': ['count', 'mean'],
            'sell_through_rate': 'mean',
            'SOLD_AMOUNT': 'sum'
        }).reset_index()
        concurrent.columns = ['fecha', 'dominio', 'num_ofertas', 'tasa_conv', 
                              'str_rate', 'gmv_total']
        return concurrent
    
    @staticmethod
    def compare_solo_vs_competition(df: pd.DataFrame, verbose: bool = True) -> Dict:
        """
        Compara ofertas solitarias vs ofertas con competencia.
        Analiza cómo la cantidad de ofertas simultáneas afecta la conversión.
        """
        concurrent = AnalisisCanibalizacion.get_concurrent_offers(df)
        
        solo = concurrent[concurrent['num_ofertas'] == 1]
        with_comp = concurrent[concurrent['num_ofertas'] > 1]
        
        # Análisis por bandas de competencia
        bins = [1, 2, 5, 10, 20, 50, float('inf')]
        labels = ['1 (sola)', '2-4', '5-9', '10-19', '20-49', '50+']
        concurrent['banda'] = pd.cut(concurrent['num_ofertas'], bins=bins, labels=labels, right=False)
        
        bandas_stats = concurrent.groupby('banda', observed=True).agg({
            'tasa_conv': ['mean', 'std', 'count'],
            'str_rate': 'mean',
            'gmv_total': ['mean', 'sum']
        }).round(4)
        bandas_stats.columns = ['conv_avg', 'conv_std', 'n_casos', 'str_avg', 'gmv_avg', 'gmv_total']
        
        # Análisis por dominio
        domain_impact = concurrent.groupby('dominio').apply(
            lambda x: pd.Series({
                'correlacion_conv_ofertas': x['num_ofertas'].corr(x['tasa_conv']),
                'ofertas_promedio': x['num_ofertas'].mean(),
                'conv_promedio': x['tasa_conv'].mean(),
                'n_dias': len(x)
            }), include_groups=False
        ).sort_values('correlacion_conv_ofertas')
        
        # Calcular métricas
        solo_conv = float(solo['tasa_conv'].mean()) if len(solo) > 0 else 0
        comp_conv = float(with_comp['tasa_conv'].mean()) if len(with_comp) > 0 else 0
        diff_pp = (solo_conv - comp_conv) * 100
        max_concurrent = int(concurrent['num_ofertas'].max())
        
        # Correlación global
        corr_global = concurrent['num_ofertas'].corr(concurrent['tasa_conv'])
        
        # Top casos con más ofertas simultáneas
        top_concurrent = concurrent.nlargest(10, 'num_ofertas')[
            ['fecha', 'dominio', 'num_ofertas', 'tasa_conv', 'str_rate', 'gmv_total']
        ].copy()
        top_concurrent['tasa_conv'] = top_concurrent['tasa_conv'].apply(lambda x: f"{x:.1%}")
        top_concurrent['str_rate'] = top_concurrent['str_rate'].apply(lambda x: f"{x:.1%}")
        top_concurrent['gmv_total'] = top_concurrent['gmv_total'].apply(lambda x: f"${x:,.0f}")
        
        result = {
            'solo_count': len(solo),
            'solo_conv_avg': solo_conv,
            'comp_count': len(with_comp),
            'comp_conv_avg': comp_conv,
            'diff_pp': diff_pp,
            'max_concurrent': max_concurrent,
            'correlacion_global': float(corr_global),
            'bandas_stats': bandas_stats,
            'domain_impact': domain_impact,
            'top_concurrent': top_concurrent
        }
        
        if verbose:
            w = 70  # ancho de la tabla
            print("\n" + "═" * w)
            print("🔄 ANÁLISIS DE CANIBALIZACIÓN: OFERTAS SOLITARIAS vs CON COMPETENCIA")
            print("═" * w)
            
            # Resumen principal
            print("\n📊 RESUMEN COMPARATIVO")
            print("─" * w)
            print(f"{'Tipo':<30} {'Casos':>12} {'Conversión':>12} {'% del Total':>12}")
            print("─" * w)
            total = len(solo) + len(with_comp)
            print(f"{'🎯 Ofertas solitarias':<30} {len(solo):>12,} {solo_conv:>11.1%} {len(solo)/total:>11.1%}")
            print(f"{'⚔️  Ofertas c/competencia':<30} {len(with_comp):>12,} {comp_conv:>11.1%} {len(with_comp)/total:>11.1%}")
            print("─" * w)
            
            # Interpretación
            if diff_pp > 2:
                emoji, msg = "⚠️", f"La competencia interna REDUCE conversión en {abs(diff_pp):.1f}pp"
            elif diff_pp < -2:
                emoji, msg = "✅", f"La competencia interna MEJORA conversión en {abs(diff_pp):.1f}pp"
            else:
                emoji, msg = "➖", f"Sin diferencia significativa ({diff_pp:+.1f}pp)"
            print(f"\n{emoji} CONCLUSIÓN: {msg}")
            print(f"📈 Correlación global (ofertas vs conversión): {corr_global:+.3f}")
            
            # Análisis por bandas
            print("\n" + "─" * w)
            print("📋 CONVERSIÓN POR NIVEL DE COMPETENCIA")
            print("─" * w)
            print(f"{'Banda':<15} {'Casos':>10} {'Conv Avg':>12} {'Conv Std':>12} {'GMV Avg':>15}")
            print("─" * w)
            for banda in bandas_stats.index:
                row = bandas_stats.loc[banda]
                print(f"{str(banda):<15} {int(row['n_casos']):>10,} {row['conv_avg']:>11.1%} "
                      f"{row['conv_std']:>11.2%} ${row['gmv_avg']:>13,.0f}")
            
            # Dominios más afectados
            print("\n" + "─" * w)
            print("🎯 IMPACTO POR DOMINIO (correlación ofertas vs conversión)")
            print("─" * w)
            print(f"{'Dominio':<25} {'Correlación':>12} {'Ofertas Avg':>12} {'Conv Avg':>12}")
            print("─" * w)
            for dom in domain_impact.index[:5]:  # Top 5 con mayor impacto negativo
                row = domain_impact.loc[dom]
                corr_emoji = "🔴" if row['correlacion_conv_ofertas'] < -0.1 else "🟡" if row['correlacion_conv_ofertas'] < 0.1 else "🟢"
                print(f"{corr_emoji} {dom:<23} {row['correlacion_conv_ofertas']:>+11.3f} "
                      f"{row['ofertas_promedio']:>11.1f} {row['conv_promedio']:>11.1%}")
            if len(domain_impact) > 5:
                print(f"   ... y {len(domain_impact) - 5} dominios más")
            
            # Top días con más competencia
            print("\n" + "─" * w)
            print(f"🏆 TOP 5 DÍAS CON MÁS OFERTAS SIMULTÁNEAS (máx: {max_concurrent})")
            print("─" * w)
            print(f"{'Fecha':<12} {'Dominio':<18} {'#Ofertas':>10} {'Conv':>10} {'GMV':>15}")
            print("─" * w)
            for _, row in top_concurrent.head(5).iterrows():
                print(f"{str(row['fecha']):<12} {row['dominio']:<18} {row['num_ofertas']:>10} "
                      f"{row['tasa_conv']:>10} {row['gmv_total']:>15}")
            
            print("\n" + "═" * w + "\n")
        
        return result
    
    @staticmethod
    def get_cannibalization_by_domain(df: pd.DataFrame) -> pd.DataFrame:
        """
        Analiza la canibalización por dominio.
        """
        concurrent = AnalisisCanibalizacion.get_concurrent_offers(df)
        
        # Agrupar por número de ofertas concurrentes
        cannib = concurrent.groupby('num_ofertas').agg({
            'tasa_conv': 'mean',
            'str_rate': 'mean',
            'gmv_total': 'mean',
            'fecha': 'count'
        })
        cannib.columns = ['tasa_conv_promedio', 'str_promedio', 'gmv_promedio', 'num_casos']
        return cannib.round(4)
    
    @staticmethod
    def plot_cannibalization(df: pd.DataFrame, figsize: Tuple = (14, 10)) -> None:
        """
        Genera gráfico de análisis de canibalización.
        
        PREGUNTA: ¿La competencia interna entre ofertas afecta la conversión?
        """
        concurrent = AnalisisCanibalizacion.get_concurrent_offers(df)
        cannib = AnalisisCanibalizacion.get_cannibalization_by_domain(df)
        comparison = AnalisisCanibalizacion.compare_solo_vs_competition(df, verbose=False)
        
        # Colores consistentes
        colors = {
            'primary': '#3498db',
            'success': '#27ae60',
            'danger': '#e74c3c',
            'warning': '#f39c12',
            'neutral': '#95a5a6',
            'gradient': plt.cm.RdYlGn_r
        }
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('¿La competencia interna entre ofertas afecta la conversión?', 
                    fontsize=16, fontweight='bold', color='#1A252F', y=1.02)
        
        # ═══════════════════════════════════════════════════════════════════
        # Gráfico 1: Conversión por número de ofertas concurrentes (top 15)
        # ═══════════════════════════════════════════════════════════════════
        ax1 = axes[0, 0]
        n_bars = min(15, len(cannib))
        x_vals = cannib.index[:n_bars]
        y_vals = cannib['tasa_conv_promedio'][:n_bars] * 100
        sizes = cannib['num_casos'][:n_bars]
        
        # Color según si mejora o empeora vs baseline (1 oferta)
        baseline = cannib.loc[1, 'tasa_conv_promedio'] * 100 if 1 in cannib.index else y_vals.mean()
        bar_colors = [colors['success'] if v >= baseline else colors['danger'] for v in y_vals]
        
        bars = ax1.bar(x_vals, y_vals, color=bar_colors, edgecolor='white', linewidth=0.5)
        ax1.axhline(y=baseline, color='#2c3e50', linestyle='--', linewidth=2, label=f'Baseline (1 oferta): {baseline:.1f}%')
        
        # Añadir número de casos encima de las barras
        for bar, size in zip(bars, sizes):
            height = bar.get_height()
            ax1.annotate(f'n={int(size)}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', 
                        fontsize=7, color='#7f8c8d', rotation=45)
        
        ax1.set_xlabel('# Ofertas Concurrentes (mismo dominio/día)', fontsize=10, fontweight='bold')
        ax1.set_ylabel('Tasa de Conversión (%)', fontsize=10, fontweight='bold')
        ax1.set_title('Conversión por Nivel de Competencia', fontsize=12, fontweight='bold', pad=10)
        ax1.legend(loc='upper right', fontsize=9)
        ax1.set_ylim(0, max(y_vals) * 1.2)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # ═══════════════════════════════════════════════════════════════════
        # Gráfico 2: Comparación Solo vs Competencia (mejorado)
        # ═══════════════════════════════════════════════════════════════════
        ax2 = axes[0, 1]
        
        categories = ['Solitarias\n(1 oferta)', 'Con Competencia\n(2+ ofertas)']
        conv_values = [comparison['solo_conv_avg'] * 100, comparison['comp_conv_avg'] * 100]
        counts = [comparison['solo_count'], comparison['comp_count']]
        bar_colors = [colors['success'], colors['primary']]
        
        bars = ax2.bar(categories, conv_values, color=bar_colors, edgecolor='white', linewidth=2, width=0.6)
        
        # Añadir valores y conteos
        for bar, val, count in zip(bars, conv_values, counts):
            height = bar.get_height()
            ax2.annotate(f'{val:.1f}%\n({count:,} casos)', 
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points", ha='center', va='bottom',
                        fontsize=11, fontweight='bold', color='#2c3e50')
        
        # Añadir diferencia
        diff = comparison['diff_pp']
        diff_color = colors['danger'] if diff > 0 else colors['success']
        diff_text = f"Δ = {diff:+.1f}pp"
        ax2.annotate(diff_text, xy=(0.5, max(conv_values) * 0.5), xycoords=('axes fraction', 'data'),
                    fontsize=14, fontweight='bold', color=diff_color, ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=diff_color, alpha=0.9))
        
        ax2.set_ylabel('Tasa de Conversión (%)', fontsize=10, fontweight='bold')
        ax2.set_title('Impacto de la Competencia Interna', fontsize=12, fontweight='bold', pad=10)
        ax2.set_ylim(0, max(conv_values) * 1.3)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        # ═══════════════════════════════════════════════════════════════════
        # Gráfico 3: Scatter plot con tendencia
        # ═══════════════════════════════════════════════════════════════════
        ax3 = axes[1, 0]
        
        # Scatter con tamaño proporcional al GMV
        scatter = ax3.scatter(concurrent['num_ofertas'], concurrent['tasa_conv'] * 100,
                             c=concurrent['gmv_total'], cmap='YlOrRd', 
                             alpha=0.6, s=30, edgecolors='white', linewidth=0.3)
        
        # Línea de tendencia
        z = np.polyfit(concurrent['num_ofertas'], concurrent['tasa_conv'] * 100, 1)
        p = np.poly1d(z)
        x_line = np.linspace(concurrent['num_ofertas'].min(), concurrent['num_ofertas'].max(), 100)
        ax3.plot(x_line, p(x_line), color=colors['danger'], linewidth=2, linestyle='--', 
                label=f'Tendencia (r={comparison["correlacion_global"]:+.3f})')
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax3, shrink=0.8)
        cbar.set_label('GMV Total ($)', fontsize=9)
        
        ax3.set_xlabel('# Ofertas Concurrentes', fontsize=10, fontweight='bold')
        ax3.set_ylabel('Tasa de Conversión (%)', fontsize=10, fontweight='bold')
        ax3.set_title('Dispersión: Competencia vs Conversión', fontsize=12, fontweight='bold', pad=10)
        ax3.legend(loc='upper right', fontsize=9)
        ax3.grid(alpha=0.3, linestyle='--')
        
        # ═══════════════════════════════════════════════════════════════════
        # Gráfico 4: Heatmap por dominio (correlación)
        # ═══════════════════════════════════════════════════════════════════
        ax4 = axes[1, 1]
        
        domain_impact = comparison['domain_impact'].copy()
        domain_impact = domain_impact.sort_values('correlacion_conv_ofertas')
        
        # Crear heatmap horizontal
        domains = domain_impact.index[:10]  # Top 10 dominios
        correlations = domain_impact.loc[domains, 'correlacion_conv_ofertas'].values
        conv_avg = domain_impact.loc[domains, 'conv_promedio'].values * 100
        
        y_pos = np.arange(len(domains))
        
        # Barras de correlación
        bar_colors = [colors['danger'] if c < 0 else colors['success'] for c in correlations]
        bars = ax4.barh(y_pos, correlations, color=bar_colors, edgecolor='white', linewidth=0.5, height=0.7)
        
        # Línea vertical en 0
        ax4.axvline(x=0, color='#2c3e50', linewidth=1.5)
        
        # Labels con conversión promedio
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels([f"{d} ({c:.0f}%)" for d, c in zip(domains, conv_avg)], fontsize=9)
        
        ax4.set_xlabel('Correlación (Ofertas vs Conversión)', fontsize=10, fontweight='bold')
        ax4.set_title('Sensibilidad por Dominio', fontsize=12, fontweight='bold', pad=10)
        ax4.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Leyenda explicativa
        ax4.annotate('🔴 Negativo = más ofertas → menos conversión\n🟢 Positivo = más ofertas → más conversión',
                    xy=(0.02, 0.98), xycoords='axes fraction', fontsize=8, va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8f9fa', edgecolor='#dee2e6'))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.show()
        plt.close(fig)


# =============================================================================
# TEMÁTICA 6: ESTRATEGIA E IMPACTO (Parte 2 - Negocio)
# =============================================================================

class AnalisisNegocio:
    """Métodos para análisis avanzado de negocio."""
    
    @staticmethod
    def get_operational_risk(df: pd.DataFrame) -> Dict:
        """
        Calcula métricas de riesgo operativo.
        """
        oversell = df[df['has_oversell']]
        
        return {
            'gmv_en_riesgo': oversell['SOLD_AMOUNT'].sum(),
            'unidades_en_riesgo': oversell['oversell_qty'].sum(),
            'ofertas_problema': len(oversell),
            'pct_ofertas_problema': len(oversell) / len(df) * 100,
            'top_categories_risk': oversell.groupby('DOM_DOMAIN_AGG1')['oversell_qty'].sum().nlargest(5)
        }
    
    @staticmethod
    def get_fomo_effect(df: pd.DataFrame) -> pd.DataFrame:
        """
        Analiza el efecto FOMO (escasez).
        """
        stock_perf = df.groupby('stock_bucket').agg({
            'has_sales': 'mean',
            'is_sold_out': 'mean',
            'duration_hours': lambda x: x[df.loc[x.index, 'is_sold_out']].mean() if x[df.loc[x.index, 'is_sold_out']].any() else np.nan,
            'SOLD_AMOUNT': 'sum',
            'OFFER_START_DATE': 'count'
        })
        stock_perf.columns = ['tasa_conversion', 'tasa_sellout', 'tiempo_hasta_sellout',
                              'gmv_total', 'num_ofertas']
        return stock_perf.round(4)
    
    @staticmethod
    def get_slot_efficiency(df: pd.DataFrame) -> Dict:
        """
        Calcula la eficiencia del slot de tiempo.
        
        NOTA: 'horas_slot' es la suma de duraciones de todas las ofertas,
        NO tiempo calendario. Múltiples ofertas pueden correr en paralelo.
        """
        df_con_ventas = df[df['has_sales']]
        zombies = df[~df['has_sales']]
        
        total_gmv = df_con_ventas['SOLD_AMOUNT'].sum()
        total_slot_hours = df['duration_hours'].sum()  # Suma de duraciones (no calendario)
        total_slots = len(df)
        
        # Horas-slot desperdiciadas en zombies
        horas_slot_zombies = zombies['duration_hours'].sum()
        
        return {
            'gmv_total': total_gmv,
            'total_slots': total_slots,
            'total_horas_slot': total_slot_hours,
            'gmv_por_slot': total_gmv / total_slots,
            'gmv_por_hora_slot': total_gmv / total_slot_hours,
            'horas_slot_zombies': horas_slot_zombies,
            'pct_horas_zombies': horas_slot_zombies / total_slot_hours * 100,
            'gmv_potencial_recuperado': (horas_slot_zombies / total_slot_hours) * total_gmv,
            # Métricas adicionales
            'slots_zombies': len(zombies),
            'slots_exitosos': len(df_con_ventas),
            'duracion_promedio_zombie': zombies['duration_hours'].mean(),
            'duracion_promedio_exitosa': df_con_ventas['duration_hours'].mean()
        }
    
    @staticmethod
    def plot_slot_efficiency(df: pd.DataFrame, figsize: Tuple = (16, 7)) -> None:
        """
        Genera gráfico de eficiencia del slot de tiempo.
        
        PREGUNTA: ¿Qué tan eficientemente estamos usando los slots?
        
        NOTA: 'Horas-slot' es la suma de duraciones de ofertas, NO tiempo calendario.
        Múltiples ofertas pueden correr en paralelo.
        """
        efficiency = AnalisisNegocio.get_slot_efficiency(df)
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle('¿Qué tan eficientemente estamos usando los slots?', 
                    fontsize=14, fontweight='bold', color='#1A252F', y=1.02)
        
        # ═══════════════════════════════════════════════════════════
        # Gráfico 1: Distribución de Horas-Slot (Donut Chart)
        # ═══════════════════════════════════════════════════════════
        ax1 = axes[0]
        horas_productivas = efficiency['total_horas_slot'] - efficiency['horas_slot_zombies']
        sizes = [horas_productivas, efficiency['horas_slot_zombies']]
        labels = ['Exitosas', 'Zombies']
        colors = [COLORS['success'], COLORS['danger']]
        
        wedges, texts, autotexts = ax1.pie(
            sizes, labels=labels, autopct='%1.1f%%', colors=colors, 
            startangle=90, pctdistance=0.75,
            wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2),
            textprops={'fontsize': 11, 'fontweight': 'medium'},
            explode=(0, 0.05)
        )
        for autotext in autotexts:
            autotext.set_fontsize(12)
            autotext.set_fontweight('bold')
        
        ax1.text(0, 0, f'{efficiency["total_horas_slot"]:,.0f}\nhoras-slot\ntotales', 
                ha='center', va='center', fontsize=11, fontweight='bold',
                color=COLORS['dark'])
        ax1.set_title('Distribución de Horas-Slot', pad=15)
        
        # Nota explicativa y fórmula
        ax1.text(0, -1.3, 
                'Horas-slot = Σ duración de cada oferta\n(NO es tiempo calendario)',
                ha='center', va='center', fontsize=10, style='italic', color='gray')
        ax1.text(0, -1.65, 
                r'$\%_{zombies} = \frac{\sum dur_{zombies}}{\sum dur_{todas}} \times 100$',
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f0f0', alpha=0.8))
        
        # ═══════════════════════════════════════════════════════════
        # Gráfico 2: Métricas de Eficiencia (Barras horizontales)
        # ═══════════════════════════════════════════════════════════
        ax2 = axes[1]
        ax2.set_facecolor('#FAFAFA')
        
        metrics_labels = ['GMV/Slot\n($)', 'GMV/Hora-slot\n($)']
        metrics_values = [efficiency['gmv_por_slot'], efficiency['gmv_por_hora_slot']]
        
        bars = ax2.barh(metrics_labels, metrics_values, color=[COLORS['primary'], COLORS['secondary']],
                       edgecolor='white', linewidth=2, height=0.5)
        
        for bar, val in zip(bars, metrics_values):
            ax2.text(val + max(metrics_values)*0.02, bar.get_y() + bar.get_height()/2, 
                    f'${val:,.2f}', va='center', ha='left', fontsize=11, fontweight='bold')
        
        ax2.set_xlim(0, max(metrics_values) * 1.4)
        ax2.set_title('Métricas de Productividad', pad=15)
        ax2.set_xlabel('Valor ($)')
        
        # Fórmulas de productividad
        formula_text = (r'$\frac{GMV}{Slot} = \frac{GMV_{total}}{n_{ofertas}}$' + '\n\n' +
                       r'$\frac{GMV}{Hora} = \frac{GMV_{total}}{\sum duración}$')
        ax2.text(0.95, 0.5, formula_text,
                transform=ax2.transAxes, ha='right', va='center', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f0f0', alpha=0.8))
        
        # ═══════════════════════════════════════════════════════════
        # Gráfico 3: GMV Potencial Recuperable
        # ═══════════════════════════════════════════════════════════
        ax3 = axes[2]
        ax3.set_facecolor('#FAFAFA')
        
        gmv_actual = efficiency['gmv_total']
        gmv_potencial = efficiency['gmv_potencial_recuperado']
        gmv_total_potencial = gmv_actual + gmv_potencial
        
        # Barras apiladas
        bar_width = 0.6
        ax3.bar(['GMV'], gmv_actual, width=bar_width, color=COLORS['primary'], 
               label=f'GMV Actual: ${gmv_actual/1e6:.2f}M', edgecolor='white', linewidth=2)
        ax3.bar(['GMV'], gmv_potencial, bottom=gmv_actual, width=bar_width, 
               color=COLORS['success'], alpha=0.7, hatch='///',
               label=f'Potencial: +${gmv_potencial/1e6:.2f}M', edgecolor='white', linewidth=2)
        
        # Anotaciones
        ax3.text(0, gmv_actual/2, f'${gmv_actual/1e6:.2f}M\nActual', 
                ha='center', va='center', fontsize=11, fontweight='bold', color='white')
        ax3.text(0, gmv_actual + gmv_potencial/2, f'+${gmv_potencial/1e6:.2f}M\nRecuperable', 
                ha='center', va='center', fontsize=10, fontweight='bold', color=COLORS['dark'])
        
        ax3.set_ylabel('GMV ($)')
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
        ax3.set_title('GMV: Actual vs Potencial', pad=15)
        ax3.legend(loc='upper right', fontsize=9)
        ax3.set_ylim(0, gmv_total_potencial * 1.15)
        
        # Fórmula del GMV potencial
        ax3.text(0.5, -0.12, 
                r'$GMV_{pot} = \frac{horas_{zombies}}{horas_{total}} \times GMV_{actual}$',
                transform=ax3.transAxes, ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0', alpha=0.8))
        
        # Insight al pie con cálculo explícito
        pct_zombies = efficiency['pct_horas_zombies']
        fig.text(0.5, -0.08, 
                f'💡 {pct_zombies:.1f}% de horas-slot en zombies × ${gmv_actual/1e6:.2f}M = ${gmv_potencial/1e6:.2f}M potenciales', 
                ha='center', fontsize=11, style='italic', color=COLORS['dark'],
                bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['light'], alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        
        # Resumen impreso
        print()
        print("=" * 70)
        print("⏱️  EFICIENCIA DE SLOTS (Horas-Slot = Σ duración de ofertas)")
        print("=" * 70)
        print(f"Total Ofertas:             {efficiency['total_slots']:,}")
        print(f"  - Exitosas:              {efficiency['slots_exitosos']:,}")
        print(f"  - Zombies:               {efficiency['slots_zombies']:,}")
        print("-" * 70)
        print(f"Total Horas-Slot:          {efficiency['total_horas_slot']:,.0f} (suma de duraciones)")
        print(f"  - En exitosas:           {horas_productivas:,.0f} ({100-pct_zombies:.1f}%)")
        print(f"  - En zombies:            {efficiency['horas_slot_zombies']:,.0f} ({pct_zombies:.1f}%)")
        print("-" * 70)
        print(f"Duración promedio exitosa: {efficiency['duracion_promedio_exitosa']:.1f} horas")
        print(f"Duración promedio zombie:  {efficiency['duracion_promedio_zombie']:.1f} horas")
        print("-" * 70)
        print(f"GMV Total:                 ${efficiency['gmv_total']:,.0f}")
        print(f"GMV por Slot:              ${efficiency['gmv_por_slot']:,.2f}")
        print(f"GMV por Hora-Slot:         ${efficiency['gmv_por_hora_slot']:,.2f}")
        print("-" * 70)
        print(f"📈 GMV Potencial:          ${efficiency['gmv_potencial_recuperado']:,.0f}")
        print("=" * 70)
        print("NOTA: Horas-slot NO es tiempo calendario. Las ofertas pueden")
        print("      correr en paralelo, por eso la suma supera las horas del día.")
        print("=" * 70)
    
    @staticmethod
    def plot_operational_risk(df: pd.DataFrame, figsize: Tuple = (16, 6)) -> None:
        """
        Genera gráfico de riesgo operativo por overselling.
        
        PREGUNTA: ¿Cuál es el impacto del riesgo operativo por overselling?
        """
        risk = AnalisisNegocio.get_operational_risk(df)
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle('¿Cuál es el impacto del riesgo operativo por overselling?', 
                    fontsize=14, fontweight='bold', color='#1A252F', y=1.02)
        
        # ═══════════════════════════════════════════════════════════
        # Gráfico 1: Proporción de ofertas con problema (Donut)
        # ═══════════════════════════════════════════════════════════
        ax1 = axes[0]
        total_ofertas = len(df)
        ofertas_ok = total_ofertas - risk['ofertas_problema']
        sizes = [ofertas_ok, risk['ofertas_problema']]
        labels = ['Sin Problemas', 'Overselling']
        colors = [COLORS['success'], COLORS['danger']]
        
        wedges, texts, autotexts = ax1.pie(
            sizes, labels=labels, autopct='%1.1f%%', colors=colors, 
            startangle=90, pctdistance=0.75,
            wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2),
            textprops={'fontsize': 11, 'fontweight': 'medium'},
            explode=(0, 0.08)
        )
        for autotext in autotexts:
            autotext.set_fontsize(12)
            autotext.set_fontweight('bold')
        
        ax1.text(0, 0, f'{total_ofertas:,}\nofertas\ntotales', 
                ha='center', va='center', fontsize=12, fontweight='bold',
                color=COLORS['dark'])
        ax1.set_title('Ofertas con Overselling', pad=15)
        
        # ═══════════════════════════════════════════════════════════
        # Gráfico 2: GMV y Unidades en Riesgo (KPIs)
        # ═══════════════════════════════════════════════════════════
        ax2 = axes[1]
        ax2.set_facecolor('#FAFAFA')
        ax2.axis('off')
        
        # KPI boxes
        kpis = [
            ('GMV en Riesgo', f"${risk['gmv_en_riesgo']:,.0f}", COLORS['danger']),
            ('Unidades en Riesgo', f"{risk['unidades_en_riesgo']:,}", COLORS['warning']),
            ('Ofertas Problema', f"{risk['ofertas_problema']:,}", COLORS['secondary'])
        ]
        
        for i, (title, value, color) in enumerate(kpis):
            y_pos = 0.75 - i * 0.3
            # Rectángulo de fondo
            rect = plt.Rectangle((0.05, y_pos - 0.1), 0.9, 0.25, 
                                 facecolor=color, alpha=0.15, transform=ax2.transAxes,
                                 edgecolor=color, linewidth=2)
            ax2.add_patch(rect)
            # Texto
            ax2.text(0.5, y_pos + 0.02, value, transform=ax2.transAxes,
                    ha='center', va='center', fontsize=18, fontweight='bold', color=color)
            ax2.text(0.5, y_pos - 0.06, title, transform=ax2.transAxes,
                    ha='center', va='center', fontsize=11, color=COLORS['dark'])
        
        ax2.set_title('Métricas de Riesgo', pad=15)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        
        # ═══════════════════════════════════════════════════════════
        # Gráfico 3: Top Categorías con Riesgo (Barras)
        # ═══════════════════════════════════════════════════════════
        ax3 = axes[2]
        ax3.set_facecolor('#FAFAFA')
        
        top_cats = risk['top_categories_risk']
        if len(top_cats) > 0:
            y_pos = np.arange(len(top_cats))
            
            # Gradiente de colores de rojo
            colors_gradient = [plt.cm.Reds(0.4 + 0.5 * (i / len(top_cats))) for i in range(len(top_cats))]
            colors_gradient.reverse()
            
            bars = ax3.barh(y_pos, top_cats.values, color=colors_gradient,
                           edgecolor='white', linewidth=2)
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(top_cats.index, fontsize=10)
            ax3.invert_yaxis()
            
            # Agregar valores
            for bar, val in zip(bars, top_cats.values):
                ax3.text(val + max(top_cats.values)*0.02, bar.get_y() + bar.get_height()/2, 
                        f'{val:,.0f}', va='center', ha='left', fontsize=10, fontweight='bold')
            
            ax3.set_xlabel('Unidades en Overselling')
            ax3.set_xlim(0, max(top_cats.values) * 1.2)
        
        ax3.set_title('Top 5 Categorías con Mayor Riesgo', pad=15)
        
        # Insight al pie
        pct_problema = risk['pct_ofertas_problema']
        fig.text(0.5, -0.02, 
                f'⚠️ El {pct_problema:.1f}% de las ofertas tienen overselling, '
                f'poniendo en riesgo ${risk["gmv_en_riesgo"]:,.0f} en GMV y {risk["unidades_en_riesgo"]:,} unidades.', 
                ha='center', fontsize=11, style='italic', color=COLORS['danger'],
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFEBEE', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        
        # Resumen impreso
        print()
        print("=" * 60)
        print("⚠️  RIESGO OPERATIVO POR OVERSELLING")
        print("=" * 60)
        print(f"Ofertas con problema:  {risk['ofertas_problema']:,} ({pct_problema:.1f}%)")
        print(f"GMV en riesgo:         ${risk['gmv_en_riesgo']:,.0f}")
        print(f"Unidades en riesgo:    {risk['unidades_en_riesgo']:,}")
        print("-" * 60)
        print("Top categorías afectadas:")
        for cat, qty in risk['top_categories_risk'].items():
            print(f"  • {cat}: {qty:,} unidades")
        print("=" * 60)
    
    @staticmethod
    def get_productivity_metrics(df: pd.DataFrame, verbose: bool = True) -> Dict:
        """
        Calcula métricas de productividad del programa.
        Analiza el GMV actual y el potencial perdido por ofertas zombie.
        """
        con_ventas = df[df['has_sales']]
        zombies = df[~df['has_sales']]
        
        # Métricas básicas
        gmv_actual = float(con_ventas['SOLD_AMOUNT'].sum())
        gmv_promedio = float(con_ventas['SOLD_AMOUNT'].mean())
        gmv_mediana = float(con_ventas['SOLD_AMOUNT'].median())
        gmv_potencial_adicional = len(zombies) * gmv_promedio
        incremento_potencial_pct = (gmv_potencial_adicional / gmv_actual * 100) if gmv_actual > 0 else 0
        
        # Métricas por categoría
        gmv_by_domain = con_ventas.groupby('DOM_DOMAIN_AGG1')['SOLD_AMOUNT'].agg(['sum', 'mean', 'count'])
        gmv_by_domain.columns = ['gmv_total', 'gmv_promedio', 'n_ofertas']
        gmv_by_domain = gmv_by_domain.sort_values('gmv_total', ascending=False)
        
        # Zombies por categoría
        zombies_by_domain = zombies.groupby('DOM_DOMAIN_AGG1').size()
        
        # Eficiencia por dominio
        eficiencia_dominio = con_ventas.groupby('DOM_DOMAIN_AGG1').agg({
            'has_sales': 'count',
            'SOLD_AMOUNT': 'sum',
            'sell_through_rate': 'mean'
        })
        total_by_domain = df.groupby('DOM_DOMAIN_AGG1').size()
        eficiencia_dominio['total_ofertas'] = total_by_domain
        eficiencia_dominio['tasa_exito'] = eficiencia_dominio['has_sales'] / eficiencia_dominio['total_ofertas']
        eficiencia_dominio = eficiencia_dominio.sort_values('SOLD_AMOUNT', ascending=False)
        
        result = {
            'total_ofertas': len(df),
            'ofertas_exitosas': len(con_ventas),
            'slots_zombie': len(zombies),
            'tasa_exito': float(len(con_ventas) / len(df)),
            'gmv_actual': gmv_actual,
            'gmv_promedio_exitosas': gmv_promedio,
            'gmv_mediana_exitosas': gmv_mediana,
            'gmv_potencial_adicional': gmv_potencial_adicional,
            'incremento_potencial_pct': incremento_potencial_pct,
            'gmv_total_potencial': gmv_actual + gmv_potencial_adicional,
            'top_dominios_gmv': gmv_by_domain.head(5).to_dict('index'),
            'eficiencia_por_dominio': eficiencia_dominio
        }
        
        if verbose:
            w = 70
            print("\n" + "═" * w)
            print("💰 MÉTRICAS DE PRODUCTIVIDAD DEL PROGRAMA")
            print("═" * w)
            
            # Resumen general
            print("\n📊 RESUMEN GENERAL")
            print("─" * w)
            print(f"{'Métrica':<40} {'Valor':>28}")
            print("─" * w)
            print(f"{'Total de ofertas':<40} {len(df):>28,}")
            print(f"{'Ofertas exitosas (con ventas)':<40} {len(con_ventas):>28,}")
            print(f"{'Ofertas zombie (sin ventas)':<40} {len(zombies):>28,}")
            print(f"{'Tasa de éxito':<40} {len(con_ventas)/len(df):>27.1%}")
            
            # GMV
            print("\n" + "─" * w)
            print("💵 ANÁLISIS DE GMV")
            print("─" * w)
            print(f"{'GMV Total Actual':<40} ${gmv_actual:>26,.2f}")
            print(f"{'GMV Promedio por oferta exitosa':<40} ${gmv_promedio:>26,.2f}")
            print(f"{'GMV Mediana por oferta exitosa':<40} ${gmv_mediana:>26,.2f}")
            
            # Potencial
            print("\n" + "─" * w)
            print("🚀 POTENCIAL DE MEJORA")
            print("─" * w)
            print(f"{'Slots zombie (oportunidad)':<40} {len(zombies):>28,}")
            print(f"{'GMV potencial adicional':<40} ${gmv_potencial_adicional:>26,.2f}")
            print(f"{'Incremento potencial':<40} {'+' + f'{incremento_potencial_pct:.1f}%':>28}")
            print(f"{'GMV total potencial':<40} ${gmv_actual + gmv_potencial_adicional:>26,.2f}")
            
            # Top dominios
            print("\n" + "─" * w)
            print("🏆 TOP 5 DOMINIOS POR GMV")
            print("─" * w)
            print(f"{'Dominio':<25} {'GMV Total':>15} {'GMV Avg':>12} {'#Ofertas':>12}")
            print("─" * w)
            for dom in gmv_by_domain.head(5).index:
                row = gmv_by_domain.loc[dom]
                print(f"{dom:<25} ${row['gmv_total']:>13,.0f} ${row['gmv_promedio']:>10,.0f} {int(row['n_ofertas']):>12,}")
            
            # Interpretación
            print("\n" + "─" * w)
            if incremento_potencial_pct > 50:
                emoji, msg = "⚠️", f"ALTO potencial sin explotar: +{incremento_potencial_pct:.0f}% GMV posible"
            elif incremento_potencial_pct > 20:
                emoji, msg = "🔶", f"Potencial MODERADO: +{incremento_potencial_pct:.0f}% GMV recuperable"
            else:
                emoji, msg = "✅", f"Buena eficiencia: solo +{incremento_potencial_pct:.0f}% GMV por recuperar"
            print(f"{emoji} CONCLUSIÓN: {msg}")
            print("═" * w + "\n")
        
        return result
    
    @staticmethod
    def plot_productivity(df: pd.DataFrame, figsize: Tuple = (14, 6)) -> None:
        """
        Genera gráfico de productividad del programa con visualización mejorada.
        
        PREGUNTA: ¿Cuánto GMV adicional se podría generar eliminando zombies?
        """
        productivity = AnalisisNegocio.get_productivity_metrics(df, verbose=False)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('¿Cuánto GMV adicional se podría generar eliminando zombies?', 
                    fontsize=14, fontweight='bold', color='#1A252F', y=1.02)
        
        # Gráfico 1: GMV actual vs potencial (stacked bar horizontal)
        gmv_actual = productivity['gmv_actual']
        gmv_potencial = productivity['gmv_potencial_adicional']
        gmv_total = gmv_actual + gmv_potencial
        
        # Barra apilada horizontal
        ax1 = axes[0]
        bar_height = 0.5
        
        # Barra de GMV actual
        ax1.barh(['GMV'], gmv_actual, height=bar_height, color=COLORS['primary'], 
                label=f'GMV Actual: ${gmv_actual/1e6:.1f}M', edgecolor='white', linewidth=2)
        # Barra de GMV potencial
        ax1.barh(['GMV'], gmv_potencial, left=gmv_actual, height=bar_height, 
                color=COLORS['success'], alpha=0.7,
                label=f'Potencial Adicional: ${gmv_potencial/1e6:.1f}M (+{productivity["incremento_potencial_pct"]:.0f}%)', 
                edgecolor='white', linewidth=2, hatch='///')
        
        # Anotaciones dentro de las barras
        ax1.text(gmv_actual/2, 0, f'${gmv_actual/1e6:.1f}M\n(Actual)', 
                ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        ax1.text(gmv_actual + gmv_potencial/2, 0, f'+${gmv_potencial/1e6:.1f}M\n(Potencial)', 
                ha='center', va='center', fontsize=11, fontweight='bold', color=COLORS['dark'])
        
        ax1.set_xlabel('GMV ($)', fontweight='medium')
        ax1.set_title('Oportunidad de Mejora: GMV Actual vs Potencial', pad=15)
        ax1.legend(loc='upper right', framealpha=0.9)
        ax1.set_xlim(0, gmv_total * 1.1)
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.0f}M'))
        ax1.set_yticks([])
        
        # Fórmula del GMV potencial
        ax1.text(0.5, -0.18, 
                r'$GMV_{potencial} = n_{zombies} \times \overline{GMV}_{exitosas}$',
                transform=ax1.transAxes, ha='center', va='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0', alpha=0.8))
        
        # Gráfico 2: Donut chart de slots
        slots_con_ventas = len(df[df['has_sales']])
        slots_zombies = len(df[~df['has_sales']])
        slots = [slots_con_ventas, slots_zombies]
        labels = ['Con Ventas', 'Zombies']
        colors = [COLORS['success'], COLORS['danger']]
        
        # Crear donut chart
        wedges, texts, autotexts = axes[1].pie(
            slots, labels=labels, autopct='%1.1f%%', colors=colors, 
            startangle=90, pctdistance=0.75,
            wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2),
            textprops={'fontsize': 11, 'fontweight': 'medium'},
            explode=(0, 0.05)  # Destacar zombies
        )
        for autotext in autotexts:
            autotext.set_fontsize(12)
            autotext.set_fontweight('bold')
        
        # Texto central
        total_slots = slots_con_ventas + slots_zombies
        axes[1].text(0, 0, f'{total_slots:,}\nslots\ntotales', 
                    ha='center', va='center', fontsize=13, fontweight='bold',
                    color=COLORS['dark'])
        axes[1].set_title('Distribución de Slots\n(Oportunidad en Zombies)', pad=15)
        
        # Fórmula debajo del donut
        axes[1].text(0, -1.4, 
                r'$\%_{zombies} = \frac{n_{zombies}}{n_{total}} \times 100$',
                ha='center', va='center', fontsize=10, style='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0', alpha=0.8))
        
        # Agregar anotación con insight y valores
        gmv_promedio = productivity['gmv_promedio_exitosas']
        fig.text(0.5, -0.06, 
                f'💡 {slots_zombies:,} zombies × ${gmv_promedio:,.0f} (avg) = +${gmv_potencial/1e6:.1f}M potenciales', 
                ha='center', fontsize=11, style='italic', color=COLORS['dark'],
                bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['light'], alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        
        # Imprimir resumen
        print()
        print("=" * 60)
        print("💰 PRODUCTIVIDAD DEL PROGRAMA")
        print("=" * 60)
        print(f"GMV Actual:              ${productivity['gmv_actual']:,.0f}")
        print(f"GMV promedio exitosas:   ${productivity['gmv_promedio_exitosas']:,.2f}")
        print(f"Slots zombie:            {productivity['slots_zombie']:,}")
        print("-" * 60)
        print(f"📈 GMV Potencial Adicional: ${productivity['gmv_potencial_adicional']:,.0f}")
        print(f"📈 Incremento potencial:    {productivity['incremento_potencial_pct']:.1f}%")
        print("-" * 60)
        print("💡 Si eliminamos zombies y los reemplazamos por ofertas promedio,")
        print(f"   podríamos incrementar el GMV en ${productivity['gmv_potencial_adicional']:,.0f}")
        print("=" * 60)


# =============================================================================
# TEMÁTICA 6: ESTRATEGIA E IMPACTO (Parte 3 - Derivado)
# =============================================================================
# Preguntas derivadas: COVID, dominios tóxicos, modelo predictivo
# =============================================================================

class AnalisisDerivado:
    """Métodos para responder preguntas derivadas del análisis inicial."""
    
    @staticmethod
    def analyze_fs_as_symptom(df: pd.DataFrame) -> Dict:
        """
        Analiza si Free Shipping es síntoma o causa del fracaso.
        """
        shipping_domain = AnalisisEnvio.get_shipping_by_domain(df, min_offers=10)
        
        fs_helps = (shipping_domain['diff_conv'] > 0).sum()
        fs_hurts = (shipping_domain['diff_conv'] < 0).sum()
        
        return {
            'dominios_analizados': len(shipping_domain),
            'dominios_fs_ayuda': fs_helps,
            'dominios_fs_perjudica': fs_hurts,
            'pct_fs_perjudica': fs_hurts / len(shipping_domain) * 100,
            'top_fs_ayuda': shipping_domain.nlargest(10, 'diff_conv'),
            'top_fs_perjudica': shipping_domain.nsmallest(10, 'diff_conv')
        }
    
    @staticmethod
    def identify_toxic_domains(df: pd.DataFrame, min_offers: int = 20,
                                max_success: float = 0.2) -> pd.DataFrame:
        """
        Identifica dominios "tóxicos" que deberían eliminarse.
        """
        return AnalisisCategoria.get_problematic_domains(df, min_offers, max_success)
    
    @staticmethod
    def analyze_covid_distortion(df: pd.DataFrame) -> Dict:
        """
        Analiza la distorsión causada por productos COVID.
        """
        covid_domains = ['MLM-SURGICAL_AND_INDUSTRIAL_MASKS', 'MLM-DISPOSABLE_GLOVES',
                         'MLM-OXIMETERS', 'MLM-THERMOMETERS']
        
        covid_df = df[df['DOM_DOMAIN_AGG1'].isin(covid_domains)]
        non_covid_df = df[~df['DOM_DOMAIN_AGG1'].isin(covid_domains)]
        
        return {
            'gmv_covid': covid_df['SOLD_AMOUNT'].sum(),
            'gmv_non_covid': non_covid_df['SOLD_AMOUNT'].sum(),
            'pct_gmv_covid': covid_df['SOLD_AMOUNT'].sum() / df['SOLD_AMOUNT'].sum() * 100,
            'conv_covid': covid_df['has_sales'].mean(),
            'conv_non_covid': non_covid_df['has_sales'].mean(),
            'ofertas_covid': len(covid_df),
            'ofertas_non_covid': len(non_covid_df)
        }
    
    @staticmethod
    def analyze_baby_products(df: pd.DataFrame) -> Dict:
        """
        Análisis específico de productos de bebé.
        """
        baby_domains = df[df['DOM_DOMAIN_AGG1'].str.contains('BABY', na=False)]
        
        return {
            'total_ofertas': len(baby_domains),
            'tasa_zombie': (~baby_domains['has_sales']).mean(),
            'gmv_total': baby_domains['SOLD_AMOUNT'].sum(),
            'ticket_promedio': baby_domains['avg_ticket'].mean(),
            'dominios': baby_domains['DOM_DOMAIN_AGG1'].unique().tolist(),
            'perf_by_domain': baby_domains.groupby('DOM_DOMAIN_AGG1').agg({
                'has_sales': 'mean',
                'SOLD_AMOUNT': 'sum'
            })
        }
    
    @staticmethod
    def get_predictive_features_correlation(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula correlaciones para modelo predictivo.
        """
        features = ['INVOLVED_STOCK', 'start_hour', 'day_of_week', 
                    'has_free_shipping', 'duration_hours']
        
        df_encoded = df.copy()
        df_encoded['has_free_shipping'] = df_encoded['has_free_shipping'].astype(int)
        df_encoded['has_sales_int'] = df_encoded['has_sales'].astype(int)
        
        correlations = df_encoded[features + ['has_sales_int']].corr()['has_sales_int'].drop('has_sales_int')
        return correlations.sort_values(ascending=False)
    
    @staticmethod
    def plot_covid_impact(df: pd.DataFrame, figsize: Tuple = (12, 5)) -> None:
        """
        Genera gráfico del impacto COVID.
        
        PREGUNTA: ¿Cuánto distorsionan los productos COVID los resultados?
        """
        covid_analysis = AnalisisDerivado.analyze_covid_distortion(df)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('¿Cuánto distorsionan los productos COVID los resultados?', 
                    fontsize=14, fontweight='bold', color='#1A252F', y=1.02)
        
        # Gráfico 1: GMV COVID vs No-COVID
        labels = ['Productos COVID', 'Otros Productos']
        values = [covid_analysis['gmv_covid'], covid_analysis['gmv_non_covid']]
        colors = ['#e74c3c', '#3498db']
        
        axes[0].bar(labels, values, color=colors)
        axes[0].set_ylabel('GMV ($)')
        axes[0].set_title(f'GMV: COVID representa {covid_analysis["pct_gmv_covid"]:.1f}%')
        
        # Gráfico 2: Conversión
        conv_values = [covid_analysis['conv_covid'] * 100, covid_analysis['conv_non_covid'] * 100]
        axes[1].bar(labels, conv_values, color=colors)
        axes[1].set_ylabel('Tasa de Conversión (%)')
        axes[1].set_title('Conversión: COVID vs Otros')
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        
        # Imprimir resumen
        diff_conv = (covid_analysis['conv_covid'] - covid_analysis['conv_non_covid']) * 100
        print()
        print("=" * 60)
        print("🦠 IMPACTO PRODUCTOS COVID")
        print("=" * 60)
        print(f"{'Métrica':<25} {'COVID':>15} {'Otros':>15}")
        print("-" * 60)
        print(f"{'Ofertas':<25} {covid_analysis['ofertas_covid']:>15,} {covid_analysis['ofertas_non_covid']:>15,}")
        print(f"{'GMV':<25} ${covid_analysis['gmv_covid']:>14,.0f} ${covid_analysis['gmv_non_covid']:>14,.0f}")
        print(f"{'Tasa Conversión':<25} {covid_analysis['conv_covid']:>14.1%} {covid_analysis['conv_non_covid']:>14.1%}")
        print("-" * 60)
        print(f"📊 Productos COVID representan {covid_analysis['pct_gmv_covid']:.1f}% del GMV total")
        if diff_conv > 0:
            print(f"✅ COVID tiene {diff_conv:.1f}pp MAYOR conversión que otros productos")
        else:
            print(f"⚠️ COVID tiene {abs(diff_conv):.1f}pp MENOR conversión que otros productos")
        print("=" * 60)


# =============================================================================
# FUNCIONES AUXILIARES Y DE EXPORTACIÓN
# =============================================================================

# Nota: save_all_plots fue removida porque los métodos de plotting ahora 
# muestran directamente con plt.show() para evitar duplicación en notebooks.


def generate_full_report(df: pd.DataFrame) -> Dict:
    """
    Genera un reporte completo con todas las métricas.
    """
    report = {
        'performance_general': PerformanceGeneral.get_success_rates(df),
        'hourly_performance': AnalisisTemporal.get_hourly_performance(df),
        'daily_performance': AnalisisTemporal.get_daily_performance(df),
        'vertical_performance': AnalisisCategoria.get_vertical_performance(df),
        'pareto_analysis': AnalisisCategoria.get_pareto_analysis(df),
        'shipping_performance': AnalisisEnvio.get_shipping_performance(df),
        'stock_performance': AnalisisStock.get_stock_performance(df),
        'zombie_summary': AnalisisZombies.get_zombie_summary(df),
        'velocity_stats': AnalisisVelocidad.get_velocity_stats(df),
        'cannibalization': AnalisisCanibalizacion.compare_solo_vs_competition(df),
        'productivity': AnalisisNegocio.get_productivity_metrics(df),
        'covid_distortion': AnalisisDerivado.analyze_covid_distortion(df)
    }
    
    return report


# =============================================================================
# EJEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    # Ejemplo de uso
    print("Cargando datos...")
    eda = OfertasEDA('ofertas_relampago.csv')
    df = eda.df
    
    print("\n" + "=" * 60)
    print("RESUMEN DEL EDA")
    print("=" * 60)
    
    # Performance general
    print("\n📊 Performance General:")
    perf = PerformanceGeneral.get_success_rates(df)
    print(f"   - Tasa de conversión: {perf['tasa_conversion']*100:.1f}%")
    print(f"   - Tasa de zombies: {perf['tasa_zombie']*100:.1f}%")
    print(f"   - Tasa de sellout: {perf['tasa_sellout']*100:.1f}%")
    
    # Productividad
    print("\n💰 Productividad:")
    prod = AnalisisNegocio.get_productivity_metrics(df)
    print(f"   - GMV actual: ${prod['gmv_actual']:,.2f}")
    print(f"   - GMV potencial adicional: ${prod['gmv_potencial_adicional']:,.2f}")
    print(f"   - Incremento potencial: {prod['incremento_potencial_pct']:.1f}%")
    
    print("\n✅ Para ver gráficos, usa los métodos plot_* de cada clase")
    print("   Ejemplo: PerformanceGeneral.plot_success_rates(df)")
