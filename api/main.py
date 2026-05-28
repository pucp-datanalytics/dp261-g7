import os
import hashlib
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, Field
from typing import List

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
model = None

# Helper to load model on demand or at startup
def load_saved_model():
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            try:
                import sys
                sys.path.append(os.path.abspath("src"))
                sys.path.append(os.path.abspath("../src"))
                sys.path.append(os.path.abspath("."))
                sys.path.append(os.path.abspath(".."))
                from preprocessing import Winsorizer  # Ensure custom Winsorizer is registered
                model = joblib.load(MODEL_PATH)
                print(f"Model loaded successfully from: {MODEL_PATH}")
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print(f"Warning: Model file not found at: {MODEL_PATH}")
    return model

# Load at startup if file exists
load_saved_model()

def get_model():
    current_model = load_saved_model()
    if current_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded or not generated yet. Please run the notebooks first."
        )
    return current_model

# Function to calculate the hash of the model
def get_model_hash():
    if os.path.exists(MODEL_PATH):
        try:
            h = hashlib.sha256()
            with open(MODEL_PATH, "rb") as f:
                h.update(f.read())
            return h.hexdigest()[:12]
        except Exception:
            return "unknown"
    return "not_found"

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
    employee_id: int = Field(..., description="ID del empleado evaluado")
    proba: float = Field(..., description="Probabilidad calculada de attrition (0.0 a 1.0)")
    label: int = Field(..., description="Clase predicha (1 si proba >= 0.35, de lo contrario 0)")
    api_version: str = Field(..., description="Versión de la API REST")
    model_version: str = Field(..., description="Versión del modelo utilizado")

class BulkFeatures(BaseModel):
    employees: List[Features]

class BulkPredictResponse(BaseModel):
    predictions: List[PredictResponse]

# Helper function to preprocess inputs and calculate synthetic features
def preprocess_input(df_input: pd.DataFrame) -> pd.DataFrame:
    df_res = df_input.copy()
    
    # 1. Ratio Salario / Edad
    df_res["ratio_salario_edad"] = df_res["monthly_salary"] / df_res["age"]
    
    # 2. Antigüedad x Satisfacción
    df_res["antiguedad_satisfaccion"] = df_res["years_at_company"] * df_res["job_satisfaction"]
    
    # 3. Rango Edad
    df_res["rango_edad"] = pd.cut(
        df_res["age"],
        bins=[0, 30, 45, 60, 100],
        labels=["joven", "adulto", "maduro", "senior"]
    ).astype(str)
    
    # 4. Drop employee_id
    if "employee_id" in df_res.columns:
        df_res = df_res.drop(columns=["employee_id"])
        
    return df_res

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
        "model_sha": get_model_hash()
    }

@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(verify_key)])
def predict(x: Features, current_model = Depends(get_model)):
    """
    Recibe las características de un empleado y calcula la probabilidad de deserción (attrition).
    Retorna la probabilidad e indicador binario basado en el umbral óptimo de 0.35.
    """
    try:
        data_dict = x.dict()
        df_input = pd.DataFrame([data_dict])
        
        # Preprocesar entrada (añadir features sintéticos, remover employee_id)
        df_processed = preprocess_input(df_input)
        
        # Realizar predicción con el Pipeline
        if hasattr(current_model, "predict_proba"):
            proba = current_model.predict_proba(df_processed)[0, 1]
        else:
            pred = current_model.predict(df_processed)[0]
            proba = float(pred)
        
        label = 1 if proba >= 0.35 else 0
        
        return PredictResponse(
            employee_id=x.employee_id,
            proba=float(proba),
            label=label,
            api_version=API_VERSION,
            model_version=MODEL_VERSION
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error durante el proceso de inferencia: {str(e)}")

@app.post("/predict_bulk", response_model=BulkPredictResponse, dependencies=[Depends(verify_key)])
def predict_bulk(x: BulkFeatures, current_model = Depends(get_model)):
    """
    Recibe un listado de empleados y calcula la probabilidad de deserción para cada uno.
    """
    try:
        if not x.employees:
            return BulkPredictResponse(predictions=[])
            
        data_list = [emp.dict() for emp in x.employees]
        df_input = pd.DataFrame(data_list)
        
        # Guardar IDs antes de quitarlos
        employee_ids = df_input["employee_id"].tolist()
        
        # Preprocesar entrada
        df_processed = preprocess_input(df_input)
        
        # Predecir probabilidades
        if hasattr(current_model, "predict_proba"):
            probs = current_model.predict_proba(df_processed)[:, 1]
        else:
            preds = current_model.predict(df_processed)
            probs = [float(p) for p in preds]
            
        predictions = []
        for emp_id, proba in zip(employee_ids, probs):
            label = 1 if proba >= 0.35 else 0
            predictions.append(
                PredictResponse(
                    employee_id=emp_id,
                    proba=float(proba),
                    label=label,
                    api_version=API_VERSION,
                    model_version=MODEL_VERSION
                )
            )
            
        return BulkPredictResponse(predictions=predictions)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predicción masiva: {str(e)}")
