"""src/models.py
Módulo para gestionar modelos entrenados del Sprint 3.
Incluye carga, predicción y utilidades para experimentos.
"""

import joblib
import pandas as pd
from pathlib import Path

# Rutas base
BASE_MODELS_DIR = Path("../models")
BASE_DATA_DIR = Path("../data/interim")


def cargar_modelo(nombre: str, ruta_base=BASE_MODELS_DIR):
    """Carga un modelo baseline desde la carpeta models.
    
    Args:
        nombre (str): Identificador del modelo ('lr', 'dt', 'rf', 'svm', 'knn')
        ruta_base (Path): Carpeta donde están los modelos
    
    Returns:
        modelo: Objeto cargado con joblib
    """
    archivo = ruta_base / f"baseline_{nombre}.pkl"
    if not archivo.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {archivo}")
    return joblib.load(archivo)


def cargar_pipeline(ruta=BASE_MODELS_DIR / "preprocessing_pipeline.pkl"):
    """Carga el pipeline de preprocesamiento guardado en Sprint 2."""
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el pipeline: {ruta}")
    return joblib.load(ruta)


def cargar_datos_limpios():
    """Carga el dataset limpio usado en Sprint 3 (employee_performance_clean.csv)."""
    ruta = BASE_DATA_DIR / "employee_performance_clean.csv"
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el dataset: {ruta}")
    return pd.read_csv(ruta)


def predecir(modelo, X):
    """Genera predicciones a partir de un modelo y datos de entrada."""
    return modelo.predict(X)


def predecir_proba(modelo, X):
    """Genera probabilidades (si el modelo lo soporta)."""
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)
    raise AttributeError(f"El modelo {type(modelo).__name__} no tiene predict_proba")
