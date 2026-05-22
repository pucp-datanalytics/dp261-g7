import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import streamlit.components.v1 as comps
import os
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
import requests
from dotenv import load_dotenv

# ==========================================
# Variables de entorno
# ==========================================
load_dotenv("dashboard/.env")

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("API_KEY", "demo123")

# ==========================================
# Configuración de Página
# ==========================================
st.set_page_config(page_title="Dashboard del Modelo Final", layout="wide")

# ==========================================
# Estilos CSS Personalizados
# ==========================================
st.markdown("""
<style>
    .kpi-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.5);
    }
    .kpi-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4CAF50;
    }
    .kpi-title {
        font-size: 1.2rem;
        color: #FFFFFF;
    }
    [data-testid="stStatusWidget"] {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

st.title("Dashboard del Modelo Final")
st.markdown(
    "Plataforma interactiva integrada con una API REST para simular predicciones del modelo final."
)

# ==========================================
# Carga de datos de fondo para SHAP
# ==========================================
@st.cache_data
def load_background_data():
    data_path = os.path.join(os.path.dirname(__file__), "X_background.csv")
    if os.path.exists(data_path):
        df_bg = pd.read_csv(data_path)
        if "attrition" in df_bg.columns:
            df_bg = df_bg.drop(columns=["attrition"])
        return df_bg
    return None


try:
    bg_data = load_background_data()
except Exception as e:
    st.error(f"Error cargando datos de fondo: {e}")
    st.stop()

# ==========================================
# Función para consumir API
# ==========================================
@st.cache_data(ttl=60)
def predecir_api(payload: dict) -> dict:
    headers = {
        "x-api-key": API_KEY
    }

    response = requests.post(
        f"{API_URL}/predict",
        json=payload,
        headers=headers,
        timeout=5
    )

    response.raise_for_status()
    return response.json()


# ==========================================
# Sección de KPIs
# ==========================================
st.subheader("KPIs Principales")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""<div class='kpi-card'>
        <div class='kpi-title'>F1 Score (Clase 1)</div>
        <div class='kpi-value'>0.10</div>
        </div>""", unsafe_allow_html=True)

with col2:
    st.markdown("""<div class='kpi-card'>
        <div class='kpi-title'>AUC-ROC</div>
        <div class='kpi-value'>0.485</div>
        </div>""", unsafe_allow_html=True)

with col3:
    st.markdown("""<div class='kpi-card'>
        <div class='kpi-title'>Modelo</div>
        <div class='kpi-value' style='color:#2196F3; font-size: 2rem;'>API REST</div>
        </div>""", unsafe_allow_html=True)

st.divider()

# ==========================================
# Estado de la API
# ==========================================
st.subheader("Estado de la API")

try:
    health_response = requests.get(f"{API_URL}/health", timeout=5)

    if health_response.status_code == 200:
        st.success("API conectada correctamente.")
    else:
        st.warning(f"La API respondió con código {health_response.status_code}")

except Exception as e:
    st.error(f"No se pudo conectar con la API: {e}")

st.divider()

# ==========================================
# Simulador de Predicciones
# ==========================================
st.subheader("Simulador de Predicciones")
st.markdown("Sube un archivo CSV con nuevos datos para predecir mediante la API.")

uploaded = st.file_uploader("Sube un CSV", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)

    st.write("**Datos cargados (Primeras 5 filas):**")
    st.dataframe(df.head(5))

    # Separar la variable objetivo si existe
    target_col = "attrition"
    y_true = None

    if target_col in df.columns:
        y_true = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        X = df.copy()

    try:
        preds = []
        preds_proba = []
        status_list = []

        progress = st.progress(0)

        for i, (_, row) in enumerate(X.iterrows()):
            payload = row.to_dict()

            try:
                result = predecir_api(payload)

                preds.append(result.get("label"))
                preds_proba.append(result.get("proba"))
                status_list.append("ok")

            except requests.exceptions.Timeout:
                preds.append(None)
                preds_proba.append(None)
                status_list.append("timeout")

            except requests.exceptions.HTTPError as e:
                preds.append(None)
                preds_proba.append(None)
                status_list.append(f"http_error_{e.response.status_code}")

            except Exception as e:
                preds.append(None)
                preds_proba.append(None)
                status_list.append(f"error: {str(e)}")

            progress.progress((i + 1) / len(X))

        df_result = df.copy()
        df_result["Predicción"] = preds
        df_result["Probabilidad"] = preds_proba
        df_result["Estado_API"] = status_list

        st.success("Predicciones calculadas con éxito mediante la API.")
        st.dataframe(df_result[["Predicción", "Probabilidad", "Estado_API"]].head(20))

        csv_result = df_result.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Descargar predicciones",
            data=csv_result,
            file_name="predicciones_api.csv",
            mime="text/csv"
        )

        # ==========================================
        # Visualizaciones
        # ==========================================
        st.divider()
        st.subheader("Rendimiento")

        tab1, tab2 = st.tabs([
            "Integración API",
            "Métricas Globales (Si hay Target)"
        ])

        with tab1:
            st.markdown("### Validación de integración")
            st.write("""
            El dashboard ya no carga el modelo localmente. 
            Cada fila del CSV se envía al endpoint `/predict` de la API REST.
            """)

            st.write("Resumen de estados de la API:")
            st.dataframe(df_result["Estado_API"].value_counts().reset_index().rename(
                columns={
                    "index": "Estado",
                    "Estado_API": "Cantidad"
                }
            ))

            st.write("Distribución de predicciones:")
            st.dataframe(df_result["Predicción"].value_counts(dropna=False).reset_index().rename(
                columns={
                    "index": "Predicción",
                    "Predicción": "Cantidad"
                }
            ))

        with tab2:
            st.markdown("### Matriz de Confusión y Curva ROC")
            st.info(
                "Para visualizar estas métricas, el CSV subido debe contener la columna objetivo 'attrition'."
            )

            if y_true is not None:
                # Filtrar filas con predicción válida
                valid_mask = df_result["Predicción"].notna()

                y_true_valid = y_true[valid_mask]
                preds_valid = pd.Series(preds)[valid_mask].astype(int)
                preds_proba_valid = pd.Series(preds_proba)[valid_mask].astype(float)

                col_m1, col_m2 = st.columns(2)

                with col_m1:
                    cm = confusion_matrix(y_true_valid, preds_valid)
                    z = cm.tolist()
                    x_labels = ["Pred 0", "Pred 1"]
                    y_labels = ["Real 0", "Real 1"]

                    fig_cm = ff.create_annotated_heatmap(
                        z,
                        x=x_labels,
                        y=y_labels,
                        colorscale="Blues",
                        showscale=True
                    )

                    fig_cm.update_layout(
                        title_text="Matriz de Confusión (Interactiva)",
                        xaxis_title="Predicción",
                        yaxis_title="Real"
                    )

                    fig_cm["layout"]["yaxis"]["autorange"] = "reversed"

                    st.plotly_chart(fig_cm, use_container_width=True)

                with col_m2:
                    fpr, tpr, thresholds = roc_curve(y_true_valid, preds_proba_valid)
                    roc_auc = auc(fpr, tpr)

                    fig_roc = go.Figure()

                    fig_roc.add_trace(go.Scatter(
                        x=fpr,
                        y=tpr,
                        name=f"ROC curve (AUC = {roc_auc:.2f})",
                        mode="lines",
                        line=dict(color="darkorange", width=2),
                        hovertemplate="FPR: %{x:.2f}<br>TPR: %{y:.2f}<br>Threshold: %{text:.2f}",
                        text=thresholds
                    ))

                    fig_roc.add_trace(go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        name="Aleatorio",
                        mode="lines",
                        line=dict(color="navy", width=2, dash="dash")
                    ))

                    fig_roc.update_layout(
                        title="Receiver Operating Characteristic (ROC)",
                        xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate",
                        xaxis=dict(range=[0, 1], constrain="domain"),
                        yaxis=dict(range=[0, 1.05]),
                        hovermode="x unified"
                    )

                    st.plotly_chart(fig_roc, use_container_width=True)

            else:
                st.warning("No se detectó la columna objetivo 'attrition' en el CSV subido.")

    except Exception as e:
        st.error(f"Error durante la predicción: {e}")

else:
    st.info("Por favor, sube un archivo CSV para generar predicciones.")