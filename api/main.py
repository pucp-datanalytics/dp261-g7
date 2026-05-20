import os
import hashlib
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, Field

# ==========================================
# Carga de Variables de Entorno y Rutas
# ==========================================
MODEL_PATH = os.getenv("MODEL_PATH", "handoff/model/final/model.pkl")
PREPROC_PATH = os.getenv("PREPROC_PATH", "handoff/model/preproc/pipeline.pkl")
API_VERSION = "1.0.0"
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")
EXPECTED_KEY = os.getenv("API_KEY")

# ==========================================
# Carga del Modelo y Preprocesador al Iniciar
# ==========================================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"No se encontró el archivo del modelo en: {MODEL_PATH}")

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Error al cargar el modelo: {e}")

preproc = None
if os.path.exists(PREPROC_PATH):
    try:
        preproc = joblib.load(PREPROC_PATH)
    except Exception as e:
        print(f"Advertencia: No se pudo cargar el preprocesador separado: {e}")

# Función para calcular el hash del modelo
def get_model_hash():
    try:
        h = hashlib.sha256()
        with open(MODEL_PATH, "rb") as f:
            h.update(f.read())
        return h.hexdigest()[:12]
    except Exception:
        return "unknown"

model_sha = get_model_hash()

# ==========================================
# Inicialización de FastAPI
# ==========================================
app = FastAPI(
    title="API de Predicción de Deserción Laboral (Attrition)",
    description="Servicio FastAPI para predecir la probabilidad de que un empleado deje la empresa.",
    version=API_VERSION
)

# ==========================================
# Dependencia de Seguridad (API Key)
# ==========================================
def verify_key(x_api_key: str = Header(None)):
    if EXPECTED_KEY:
        if not x_api_key:
            raise HTTPException(status_code=401, detail="X-API-Key header is missing")
        if x_api_key != EXPECTED_KEY:
            raise HTTPException(status_code=401, detail="API key is invalid")

# ==========================================
# Esquemas Pydantic (Modelos de Contratos)
# ==========================================
class Features(BaseModel):
    employee_id: int = Field(..., description="ID único del empleado", example=13981)
    age: float = Field(..., description="Edad del empleado", example=51.0)
    gender: str = Field(..., description="Género (Male/Female)", example="Male")
    education: str = Field(..., description="Nivel educativo (Bachelor/Master/PhD/High School)", example="Master")
    department: str = Field(..., description="Departamento", example="IT")
    job_level: int = Field(..., description="Nivel de puesto (1-5)", example=2)
    years_at_company: float = Field(..., description="Años en la compañía", example=1.3)
    years_in_current_role: float = Field(..., description="Años en el rol actual", example=0.4)
    monthly_salary: float = Field(..., description="Salario mensual del empleado", example=4142.39)
    overtime_hours_monthly: int = Field(..., description="Horas extra mensuales promedio", example=12)
    num_projects_completed: int = Field(..., description="Número de proyectos completados", example=5)
    performance_rating: int = Field(..., description="Evaluación de desempeño", example=2)
    training_hours_yearly: float = Field(..., description="Horas de entrenamiento anuales", example=53.3)
    work_life_balance_score: int = Field(..., description="Puntuación de equilibrio vida-trabajo (1-5)", example=4)
    distance_from_home_km: float = Field(..., description="Distancia al trabajo en km", example=15.0)
    num_companies_worked: int = Field(..., description="Número de compañías en las que ha trabajado", example=3)
    job_satisfaction: int = Field(..., description="Satisfacción con el trabajo (1-5)", example=5)
    relationship_with_manager: int = Field(..., description="Relación con el manager (1-5)", example=4)
    stock_option_level: int = Field(..., description="Nivel de opciones sobre acciones (0-3)", example=1)

class PredictResponse(BaseModel):
    proba: float = Field(..., description="Probabilidad calculada de attrition (0.0 a 1.0)")
    label: int = Field(..., description="Clase predicha (1 si proba >= 0.35, de lo contrario 0)")
    api_version: str = Field(..., description="Versión de la API REST")
    model_version: str = Field(..., description="Versión del modelo utilizado")

# ==========================================
# Endpoints de la API
# ==========================================
@app.get("/health")
def health():
    """Retorna el estado de salud del servicio."""
    return {"status": "ok"}

@app.get("/version")
def version():
    """Retorna las versiones de la API, el modelo y el identificador hash único del modelo cargado."""
    return {
        "api_version": API_VERSION,
        "model_version": MODEL_VERSION,
        "model_sha": model_sha
    }

@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(verify_key)])
def predict(x: Features):
    """
    Recibe las características de un empleado y calcula la probabilidad de deserción (attrition).
    Retorna la probabilidad e indicador binario basado en el umbral óptimo de 0.35.
    """
    try:
        # Convertir datos de entrada a DataFrame con el orden y nombres de columnas adecuados
        data_dict = x.dict()
        df_input = pd.DataFrame([data_dict])
        
        # Realizar predicción con el Pipeline
        # final_model.pkl es un pipeline que ya contiene el preprocesamiento y el StackingClassifier
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df_input)[0, 1]
        else:
            # Fallback en caso de que el modelo no tenga predict_proba (ej. SVM sin probabilidad activada)
            pred = model.predict(df_input)[0]
            proba = float(pred)
        
        # Calcular etiqueta binaria según el umbral óptimo de 0.35
        label = 1 if proba >= 0.35 else 0
        
        return PredictResponse(
            proba=float(proba),
            label=label,
            api_version=API_VERSION,
            model_version=MODEL_VERSION
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error durante el proceso de inferencia: {str(e)}")
