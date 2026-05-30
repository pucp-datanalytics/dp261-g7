import os
import hashlib
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from pathlib import Path



# ==========================================
# Variables de entorno y rutas
# ==========================================
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    str(Path("../models/validated_final_model.pkl"))
)
API_VERSION = "1.0.0"
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")
EXPECTED_KEY = os.getenv("API_KEY")

# Umbral de negocio
BUSINESS_THRESHOLD = float(os.getenv("BUSINESS_THRESHOLD", "0.35"))

# Matriz costo-beneficio
BENEFIT_TP = float(os.getenv("BENEFIT_TP", "150"))
COST_FP = float(os.getenv("COST_FP", "-20"))
COST_FN = float(os.getenv("COST_FN", "-200"))
BENEFIT_TN = float(os.getenv("BENEFIT_TN", "0"))

# Rango visual del velocímetro esperado por empleado
GAUGE_MIN = float(os.getenv("GAUGE_MIN", "-200"))
GAUGE_MAX = float(os.getenv("GAUGE_MAX", "150"))


# ==========================================
# Carga del modelo
# ==========================================
model = None


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

                try:
                    from preprocessing import Winsorizer
                except Exception:
                    pass

                model = joblib.load(MODEL_PATH)
                print(f"Model loaded successfully from: {MODEL_PATH}")

            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print(f"Warning: Model file not found at: {MODEL_PATH}")

    return model


load_saved_model()


def get_model():
    current_model = load_saved_model()

    if current_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please check MODEL_PATH."
        )

    return current_model


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
# FastAPI
# ==========================================
app = FastAPI(
    title="API de Predicción de Deserción Laboral",
    description="Servicio FastAPI para predecir attrition y exponer indicadores de Business Value.",
    version=API_VERSION
)


# ==========================================
# Seguridad
# ==========================================
def verify_key(x_api_key: str = Header(None)):
    if EXPECTED_KEY:
        if not x_api_key:
            raise HTTPException(
                status_code=401,
                detail="X-API-Key header is missing"
            )

        if x_api_key != EXPECTED_KEY:
            raise HTTPException(
                status_code=401,
                detail="API key is invalid"
            )


# ==========================================
# Schemas
# ==========================================
class Features(BaseModel):
    employee_id: Optional[int] = Field(None, example=13981)
    age: float = Field(..., example=51.0)
    gender: str = Field(..., example="Male")
    education: str = Field(..., example="Master")
    department: str = Field(..., example="IT")
    job_level: int = Field(..., example=2)
    years_at_company: float = Field(..., example=1.3)
    years_in_current_role: float = Field(..., example=0.4)
    monthly_salary: float = Field(..., example=4142.39)
    overtime_hours_monthly: int = Field(..., example=12)
    num_projects_completed: int = Field(..., example=5)
    performance_rating: int = Field(..., example=2)
    training_hours_yearly: float = Field(..., example=53.3)
    work_life_balance_score: int = Field(..., example=4)
    distance_from_home_km: float = Field(..., example=15.0)
    num_companies_worked: int = Field(..., example=3)
    job_satisfaction: int = Field(..., example=5)
    relationship_with_manager: int = Field(..., example=4)
    stock_option_level: int = Field(..., example=1)


class GaugeResponse(BaseModel):
    value: float
    vmin: float
    vmax: float
    zone: str
    zones: List[Dict[str, Any]]


class PredictResponse(BaseModel):
    employee_id: int
    proba: float
    label: int
    threshold: float
    action_recommended: str
    expected_value_if_intervene: float
    expected_value_if_not_intervene: float
    expected_business_value: float
    gauge: GaugeResponse
    api_version: str
    model_version: str


class BulkFeatures(BaseModel):
    employees: List[Features]


class BulkPredictResponse(BaseModel):
    predictions: List[PredictResponse]


class BusinessConfigResponse(BaseModel):
    threshold: float
    benefit_tp: float
    cost_fp: float
    cost_fn: float
    benefit_tn: float
    gauge_min: float
    gauge_max: float
    zones: List[Dict[str, Any]]


# ==========================================
# Helpers
# ==========================================
def model_to_dict(x: BaseModel) -> dict:
    if hasattr(x, "model_dump"):
        return x.model_dump()
    return x.dict()


def preprocess_input(df_input: pd.DataFrame) -> pd.DataFrame:
    df_res = df_input.copy()

    if "ratio_salario_edad" not in df_res.columns:
        df_res["ratio_salario_edad"] = (
            df_res["monthly_salary"] / df_res["age"].replace(0, pd.NA)
        ).fillna(0)

    if "antiguedad_satisfaccion" not in df_res.columns:
        df_res["antiguedad_satisfaccion"] = (
            df_res["years_at_company"] * df_res["job_satisfaction"]
        )

    if "rango_edad" not in df_res.columns:
        df_res["rango_edad"] = pd.cut(
            df_res["age"],
            bins=[0, 30, 45, 60, 100],
            labels=["joven", "adulto", "maduro", "senior"]
        ).astype(str)

    if "employee_id" in df_res.columns:
        df_res = df_res.drop(columns=["employee_id"])

    return df_res


def get_gauge_zones():
    total_range = GAUGE_MAX - GAUGE_MIN

    return [
        {
            "name": "rojo",
            "label": "Desfavorable",
            "min": GAUGE_MIN,
            "max": GAUGE_MIN + total_range * 0.40,
            "color": "#e74c3c"
        },
        {
            "name": "amarillo",
            "label": "Intermedio",
            "min": GAUGE_MIN + total_range * 0.40,
            "max": GAUGE_MIN + total_range * 0.70,
            "color": "#f1c40f"
        },
        {
            "name": "verde",
            "label": "Favorable",
            "min": GAUGE_MIN + total_range * 0.70,
            "max": GAUGE_MAX,
            "color": "#2ecc71"
        }
    ]


def get_gauge_zone(value: float) -> str:
    zones = get_gauge_zones()

    for zone in zones:
        if zone["min"] <= value <= zone["max"]:
            return zone["name"]

    if value < GAUGE_MIN:
        return "rojo"

    return "verde"


def calculate_expected_business_value(proba: float):
    expected_value_if_intervene = (
        proba * BENEFIT_TP +
        (1 - proba) * COST_FP
    )

    expected_value_if_not_intervene = (
        proba * COST_FN +
        (1 - proba) * BENEFIT_TN
    )

    label = 1 if proba >= BUSINESS_THRESHOLD else 0

    if label == 1:
        action = "intervenir"
        expected_business_value = expected_value_if_intervene
    else:
        action = "no_intervenir"
        expected_business_value = expected_value_if_not_intervene

    gauge = {
        "value": float(expected_business_value),
        "vmin": GAUGE_MIN,
        "vmax": GAUGE_MAX,
        "zone": get_gauge_zone(expected_business_value),
        "zones": get_gauge_zones()
    }

    return {
        "label": label,
        "action_recommended": action,
        "expected_value_if_intervene": float(expected_value_if_intervene),
        "expected_value_if_not_intervene": float(expected_value_if_not_intervene),
        "expected_business_value": float(expected_business_value),
        "gauge": gauge
    }


def build_prediction_response(employee_id: int, proba: float) -> PredictResponse:
    business = calculate_expected_business_value(proba)

    return PredictResponse(
        employee_id=employee_id,
        proba=float(proba),
        label=business["label"],
        threshold=BUSINESS_THRESHOLD,
        action_recommended=business["action_recommended"],
        expected_value_if_intervene=business["expected_value_if_intervene"],
        expected_value_if_not_intervene=business["expected_value_if_not_intervene"],
        expected_business_value=business["expected_business_value"],
        gauge=business["gauge"],
        api_version=API_VERSION,
        model_version=MODEL_VERSION
    )


# ==========================================
# Endpoints
# ==========================================
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/version")
def version():
    return {
        "api_version": API_VERSION,
        "model_version": MODEL_VERSION,
        "model_sha": get_model_hash()
    }


@app.get("/business_config", response_model=BusinessConfigResponse)
def business_config():
    return BusinessConfigResponse(
        threshold=BUSINESS_THRESHOLD,
        benefit_tp=BENEFIT_TP,
        cost_fp=COST_FP,
        cost_fn=COST_FN,
        benefit_tn=BENEFIT_TN,
        gauge_min=GAUGE_MIN,
        gauge_max=GAUGE_MAX,
        zones=get_gauge_zones()
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    dependencies=[Depends(verify_key)]
)
def predict(x: Features, current_model=Depends(get_model)):
    try:
        data_dict = model_to_dict(x)

        df_input = pd.DataFrame([data_dict])
        df_processed = preprocess_input(df_input)

        if hasattr(current_model, "predict_proba"):
            proba = current_model.predict_proba(df_processed)[0, 1]
        else:
            pred = current_model.predict(df_processed)[0]
            proba = float(pred)

        emp_id = x.employee_id if x.employee_id is not None else 99999

        return build_prediction_response(
            employee_id=emp_id,
            proba=float(proba)
        )

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error durante el proceso de inferencia: {str(e)}"
        )


@app.post(
    "/predict_bulk",
    response_model=BulkPredictResponse,
    dependencies=[Depends(verify_key)]
)
def predict_bulk(x: BulkFeatures, current_model=Depends(get_model)):
    try:
        if not x.employees:
            return BulkPredictResponse(predictions=[])

        data_list = [
            model_to_dict(emp)
            for emp in x.employees
        ]

        df_input = pd.DataFrame(data_list)

        employee_ids = []

        for i, emp in enumerate(x.employees):
            if emp.employee_id is not None:
                employee_ids.append(emp.employee_id)
            else:
                employee_ids.append(i + 1)

        df_processed = preprocess_input(df_input)

        if hasattr(current_model, "predict_proba"):
            probs = current_model.predict_proba(df_processed)[:, 1]
        else:
            preds = current_model.predict(df_processed)
            probs = [float(p) for p in preds]

        predictions = []

        for emp_id, proba in zip(employee_ids, probs):
            predictions.append(
                build_prediction_response(
                    employee_id=emp_id,
                    proba=float(proba)
                )
            )

        return BulkPredictResponse(predictions=predictions)

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error en predicción masiva: {str(e)}"
        )