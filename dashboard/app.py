import streamlit as st
import pandas as pd
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

# Variables de entorno
load_dotenv("dashboard/.env")

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "mi-api-key-secreta-123")

# Configuración de Página
st.set_page_config(page_title="Dashboard de Gestión de Atrición", layout="wide")

# Estilos CSS Personalizados
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
        color: #FF4B4B;
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

st.title("Dashboard de Predicción de Deserción Laboral (Attrition)")
st.markdown(
    "Plataforma interactiva para la predicción masiva de fuga de talento mediante consumo de la API REST."
)

# Carga de datos de fondo
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

# Función para consumir API en Bulk
def predecir_api_bulk(employees_list: list) -> list:
    headers = {
        "x-api-key": API_KEY
    }
    payload = {"employees": employees_list}
    response = requests.post(
        f"{API_URL}/predict_bulk",
        json=payload,
        headers=headers,
        timeout=30
    )
    response.raise_for_status()
    return response.json().get("predictions", [])

# Sección de KPIs
st.subheader("KPIs del Modelo (Ensembles Stacking)")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""<div class='kpi-card'>
        <div class='kpi-title'>Umbral Óptimo de Decisión</div>
        <div class='kpi-value' style='color:#FFC107;'>0.35</div>
        </div>""", unsafe_allow_html=True)

with col2:
    st.markdown("""<div class='kpi-card'>
        <div class='kpi-title'>AUC-ROC (Stacking)</div>
        <div class='kpi-value' style='color:#4CAF50;'>~0.52</div>
        </div>""", unsafe_allow_html=True)

with col3:
    st.markdown("""<div class='kpi-card'>
        <div class='kpi-title'>Arquitectura del Meta-Modelo</div>
        <div class='kpi-value' style='color:#2196F3; font-size: 2rem;'>Stacking Classifier</div>
        </div>""", unsafe_allow_html=True)

st.divider()

# ==========================================
# Estado de la API
# ==========================================
st.subheader("Estado de la API REST")

try:
    health_response = requests.get(f"{API_URL}/health", timeout=5)
    if health_response.status_code == 200:
        st.success(f"API conectada correctamente en {API_URL}.")
    else:
        st.warning(f"La API respondió con código {health_response.status_code}")
except Exception as e:
    st.error(f"No se pudo conectar con la API REST en {API_URL}: {e}. Asegúrate de que el backend FastAPI esté ejecutándose.")

st.divider()

# ==========================================
# Simulador de Predicciones
# ==========================================
st.subheader("Simulador de Predicciones Masivas")
st.markdown("Sube un archivo CSV con nuevos datos para predecir mediante la API REST usando el endpoint de predicción masiva `/predict_bulk`.")

uploaded = st.file_uploader("Sube un CSV", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)

    st.write("**Datos cargados (Primeras 5 filas):**")
    st.dataframe(df.head(5))

    # Identificar columna target
    target_col = "attrition"
    y_true = None

    if target_col in df.columns:
        y_true = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        X = df.copy()

    # Si no tiene employee_id, lo agregamos en df y X para que no falle la visualización ni la validación de la API
    if "employee_id" not in df.columns:
        df["employee_id"] = np.arange(1, len(df) + 1)
    if "employee_id" not in X.columns:
        X["employee_id"] = df["employee_id"]

    # Botón para gatillar la predicción
    if st.button("Ejecutar Predicciones"):
        try:
            with st.spinner("Enviando datos al backend para predicción masiva..."):
                # Convertir dataframe a lista de registros
                employees_list = X.to_dict(orient="records")
                
                # Consumir API Bulk
                results = predecir_api_bulk(employees_list)
                
                # Extraer predicciones
                preds = [res.get("label") for res in results]
                preds_proba = [res.get("proba") for res in results]
                status_list = ["ok"] * len(results)

                df_result = df.copy()
                df_result["Predicción"] = preds
                df_result["Probabilidad"] = preds_proba
                df_result["Estado_API"] = status_list

            st.success("Predicciones procesadas con éxito por la API REST.")
            st.dataframe(df_result[["employee_id", "Predicción", "Probabilidad", "Estado_API"]].head(20))

            csv_result = df_result.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Descargar CSV de Resultados",
                data=csv_result,
                file_name="predicciones_atricion_final.csv",
                mime="text/csv"
            )

            # ==========================================
            # Visualizaciones
            # ==========================================
            st.divider()
            st.subheader("Análisis de Resultados de Predicción")

            tab1, tab2 = st.tabs([
                "Distribución de Predicciones",
                "Evaluación contra Target (Si está disponible)"
            ])

            with tab1:
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    fig_dist = go.Figure()
                    fig_dist.add_trace(go.Histogram(
                        x=df_result["Probabilidad"],
                        nbinsx=20,
                        marker_color='#FF4B4B',
                        opacity=0.75
                    ))
                    fig_dist.update_layout(
                        title="Distribución de Probabilidad de Fuga",
                        xaxis_title="Probabilidad de Atrición",
                        yaxis_title="Cantidad de Empleados",
                        bargap=0.05
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
                    
                with col_chart2:
                    pred_counts = pd.Series(preds).value_counts()
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=["Permanencia (0)", "Riesgo de Fuga (1)"],
                        values=[pred_counts.get(0, 0), pred_counts.get(1, 0)],
                        hole=.4,
                        marker_colors=['#4CAF50', '#FF4B4B']
                    )])
                    fig_pie.update_layout(title="Proporción de Empleados Predichos en Riesgo")
                    st.plotly_chart(fig_pie, use_container_width=True)

            with tab2:
                if y_true is not None:
                    col_m1, col_m2 = st.columns(2)

                    with col_m1:
                        cm = confusion_matrix(y_true, preds)
                        z = cm.tolist()
                        x_labels = ["Pred 0 (No Fuga)", "Pred 1 (Fuga)"]
                        y_labels = ["Real 0 (No Fuga)", "Real 1 (Fuga)"]

                        fig_cm = ff.create_annotated_heatmap(
                            z,
                            x=x_labels,
                            y=y_labels,
                            colorscale="Reds",
                            showscale=True
                        )

                        fig_cm.update_layout(
                            title_text="Matriz de Confusión",
                            xaxis_title="Predicción",
                            yaxis_title="Valor Real"
                        )

                        fig_cm["layout"]["yaxis"]["autorange"] = "reversed"
                        st.plotly_chart(fig_cm, use_container_width=True)

                    with col_m2:
                        fpr, tpr, thresholds = roc_curve(y_true, preds_proba)
                        roc_auc = auc(fpr, tpr)

                        fig_roc = go.Figure()

                        fig_roc.add_trace(go.Scatter(
                            x=fpr,
                            y=tpr,
                            name=f"ROC curve (AUC = {roc_auc:.4f})",
                            mode="lines",
                            line=dict(color="#FF4B4B", width=3),
                            hovertemplate="FPR: %{x:.2f}<br>TPR: %{y:.2f}<br>Threshold: %{text:.2f}",
                            text=thresholds
                        ))

                        fig_roc.add_trace(go.Scatter(
                            x=[0, 1],
                            y=[0, 1],
                            name="Aleatorio (AUC = 0.5)",
                            mode="lines",
                            line=dict(color="gray", width=2, dash="dash")
                        ))

                        fig_roc.update_layout(
                            title="Curva ROC de Validación",
                            xaxis_title="Tasa de Falsos Positivos (FPR)",
                            yaxis_title="Tasa de Verdaderos Positivos (TPR)",
                            xaxis=dict(range=[0, 1]),
                            yaxis=dict(range=[0, 1.05]),
                            hovermode="x unified"
                        )

                        st.plotly_chart(fig_roc, use_container_width=True)
                else:
                    st.warning("La columna objetivo 'attrition' no está presente en el archivo CSV subido para calcular la matriz de confusión y la curva ROC.")

        except Exception as e:
            st.error(f"Error procesando las predicciones masivas de la API: {e}")

else:
    st.info("Por favor, sube un archivo CSV estructurado con las características del empleado para iniciar.")