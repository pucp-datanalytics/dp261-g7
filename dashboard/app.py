import os
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.figure_factory as ff

from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix, roc_curve, auc


# ======================================================
# Configuración
# ======================================================
load_dotenv("dashboard/.env")

API_URL = os.getenv("API_URL", "http://184.73.43.218:8000").rstrip("/")
API_KEY = os.getenv("API_KEY", "mi-api-key-secreta-123")

HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

BUSINESS_THRESHOLD = 0.35
GAUGE_MIN = -200
GAUGE_MAX = 150


st.set_page_config(
    page_title="Dashboard de Gestión de Attrition",
    layout="wide"
)


# ======================================================
# Estilos
# ======================================================
st.markdown("""
<style>
.kpi-card {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.5);
}
.kpi-title {
    font-size: 1.05rem;
    color: #FFFFFF;
}
.kpi-value {
    font-size: 2.3rem;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# ======================================================
# Funciones API
# ======================================================
@st.cache_data(ttl=60)
def check_health(api_url: str):
    try:
        r = requests.get(f"{api_url}/health", timeout=5)
        return r.status_code, r.text
    except Exception as e:
        return None, str(e)


def predecir_api(payload: dict) -> dict:
    response = requests.post(
        f"{API_URL}/predict",
        json=payload,
        headers=HEADERS,
        timeout=10
    )
    response.raise_for_status()
    return response.json()


def limpiar_payload(row: dict) -> dict:
    payload = {}

    for k, v in row.items():
        if pd.isna(v):
            payload[k] = None
        elif isinstance(v, (np.integer,)):
            payload[k] = int(v)
        elif isinstance(v, (np.floating,)):
            payload[k] = float(v)
        else:
            payload[k] = v

    return payload


# ======================================================
# Business Value local para velocímetro
# ======================================================
def calcular_business_value(proba: float):
    benefit_tp = 150
    cost_fp = -20
    cost_fn = -200
    benefit_tn = 0

    expected_if_intervene = proba * benefit_tp + (1 - proba) * cost_fp
    expected_if_not = proba * cost_fn + (1 - proba) * benefit_tn

    if proba >= BUSINESS_THRESHOLD:
        action = "Intervenir"
        value = expected_if_intervene
    else:
        action = "No intervenir"
        value = expected_if_not

    return value, action, expected_if_intervene, expected_if_not


def crear_velocimetro(value, vmin=GAUGE_MIN, vmax=GAUGE_MAX, title="Business Value Esperado"):
    value = max(min(value, vmax), vmin)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"prefix": "USD ", "font": {"size": 34}},
        title={"text": title, "font": {"size": 18}},
        gauge={
            "axis": {"range": [vmin, vmax]},
            "bar": {"color": "black"},
            "steps": [
                {"range": [vmin, vmin + (vmax - vmin) * 0.4], "color": "#e74c3c"},
                {"range": [vmin + (vmax - vmin) * 0.4, vmin + (vmax - vmin) * 0.7], "color": "#f1c40f"},
                {"range": [vmin + (vmax - vmin) * 0.7, vmax], "color": "#2ecc71"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 4},
                "thickness": 0.75,
                "value": value
            }
        }
    ))

    fig.update_layout(
        height=320,
        margin=dict(l=30, r=30, t=60, b=20)
    )

    return fig


# ======================================================
# Header
# ======================================================
st.title("Dashboard de Predicción de Deserción Laboral")
st.markdown("Dashboard integrado con API REST desplegada en AWS para simular predicciones de attrition.")


# ======================================================
# KPIs superiores
# ======================================================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-title'>Umbral de Decisión</div>
        <div class='kpi-value' style='color:#FFC107;'>{BUSINESS_THRESHOLD:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='kpi-card'>
        <div class='kpi-title'>Endpoint API</div>
        <div class='kpi-value' style='color:#4CAF50; font-size:1.5rem;'>/predict</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='kpi-card'>
        <div class='kpi-title'>Modo de Predicción</div>
        <div class='kpi-value' style='color:#2196F3; font-size:1.5rem;'>Fila por fila</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()


# ======================================================
# Estado API
# ======================================================
st.subheader("Estado de la API REST")

status_code, health_msg = check_health(API_URL)

if status_code == 200:
    st.success(f"API conectada correctamente en {API_URL}")
else:
    st.error(f"No se pudo conectar correctamente con la API. Detalle: {health_msg}")

st.divider()


# ======================================================
# Simulador
# ======================================================
st.subheader("Simulador de Predicciones")

uploaded = st.file_uploader("Sube un CSV", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)

    st.write("**Datos cargados:**")
    st.dataframe(df.head(5))

    target_col = "attrition"

    if target_col in df.columns:
        y_true = df[target_col].copy()
        X = df.drop(columns=[target_col]).copy()
    else:
        y_true = None
        X = df.copy()

    if "employee_id" not in X.columns:
        X["employee_id"] = np.arange(1, len(X) + 1)

    if st.button("Ejecutar Predicciones"):
        results = []
        errores = []

        progress = st.progress(0)
        status_text = st.empty()

        start = time.time()

        for i, row in X.iterrows():
            try:
                payload = limpiar_payload(row.to_dict())
                res = predecir_api(payload)
                results.append(res)
            except Exception as e:
                errores.append({
                    "fila": i,
                    "error": str(e)
                })

            progress.progress((i + 1) / len(X))
            status_text.text(f"Procesando {i + 1} de {len(X)} registros...")

        elapsed = time.time() - start

        if errores:
            st.error(f"Se encontraron {len(errores)} errores durante la predicción.")
            st.dataframe(pd.DataFrame(errores).head(20))

        if results:
            df_result = df.copy()

            df_result["Predicción"] = [r.get("label") for r in results]
            df_result["Probabilidad"] = [r.get("proba") for r in results]

            df_result["Business_Value"] = [
                r.get("expected_business_value", calcular_business_value(r.get("proba", 0))[0])
                for r in results
            ]

            df_result["Acción_Recomendada"] = [
                r.get("action_recommended", calcular_business_value(r.get("proba", 0))[1])
                for r in results
            ]

            st.success(f"Predicciones procesadas correctamente en {elapsed:.2f} segundos.")

            st.dataframe(
                df_result[
                    ["employee_id", "Predicción", "Probabilidad", "Acción_Recomendada", "Business_Value"]
                ].head(30)
            )

            csv_result = df_result.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Descargar CSV de resultados",
                data=csv_result,
                file_name="predicciones_attrition_api.csv",
                mime="text/csv"
            )

            st.divider()
            st.subheader("Resumen de Resultados")

            total = len(df_result)
            riesgo = int(df_result["Predicción"].sum())
            pct_riesgo = riesgo / total if total > 0 else 0
            avg_proba = df_result["Probabilidad"].mean()
            total_value = df_result["Business_Value"].sum()
            avg_value = df_result["Business_Value"].mean()

            k1, k2, k3, k4 = st.columns(4)

            k1.metric("Registros procesados", f"{total:,}")
            k2.metric("Empleados en riesgo", f"{riesgo:,}", f"{pct_riesgo:.1%}")
            k3.metric("Probabilidad promedio", f"{avg_proba:.3f}")
            k4.metric("Business Value total", f"USD {total_value:,.0f}")

            st.divider()

            tab1, tab2, tab3, tab4 = st.tabs([
                "Distribución",
                "Business Value",
                "Velocímetro",
                "Evaluación"
            ])

            with tab1:
                c1, c2 = st.columns(2)

                with c1:
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(
                        x=df_result["Probabilidad"],
                        nbinsx=20
                    ))
                    fig_hist.update_layout(
                        title="Distribución de Probabilidades",
                        xaxis_title="Probabilidad de Attrition",
                        yaxis_title="Cantidad"
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

                with c2:
                    pred_counts = df_result["Predicción"].value_counts()

                    fig_pie = go.Figure(data=[
                        go.Pie(
                            labels=["No riesgo (0)", "Riesgo (1)"],
                            values=[pred_counts.get(0, 0), pred_counts.get(1, 0)],
                            hole=0.4
                        )
                    ])
                    fig_pie.update_layout(title="Proporción de Predicciones")
                    st.plotly_chart(fig_pie, use_container_width=True)

            with tab2:
                c1, c2 = st.columns(2)

                with c1:
                    fig_bv = go.Figure()
                    fig_bv.add_trace(go.Histogram(
                        x=df_result["Business_Value"],
                        nbinsx=20
                    ))
                    fig_bv.update_layout(
                        title="Distribución de Business Value Esperado",
                        xaxis_title="Business Value por empleado",
                        yaxis_title="Cantidad"
                    )
                    st.plotly_chart(fig_bv, use_container_width=True)

                with c2:
                    action_counts = df_result["Acción_Recomendada"].value_counts()

                    fig_action = go.Figure(data=[
                        go.Pie(
                            labels=action_counts.index,
                            values=action_counts.values,
                            hole=0.4
                        )
                    ])
                    fig_action.update_layout(title="Acciones recomendadas")
                    st.plotly_chart(fig_action, use_container_width=True)

            with tab3:
                st.markdown("### Velocímetro de Business Value promedio")

                fig_gauge = crear_velocimetro(
                    value=avg_value,
                    vmin=GAUGE_MIN,
                    vmax=GAUGE_MAX,
                    title="Business Value Promedio por Empleado"
                )

                st.plotly_chart(fig_gauge, use_container_width=True)

                st.info(
                    f"Business Value promedio: USD {avg_value:,.2f}. "
                    f"Business Value total: USD {total_value:,.2f}."
                )

            with tab4:
                if y_true is not None:
                    preds = df_result["Predicción"]
                    preds_proba = df_result["Probabilidad"]

                    c1, c2 = st.columns(2)

                    with c1:
                        cm = confusion_matrix(y_true, preds)

                        fig_cm = ff.create_annotated_heatmap(
                            cm.tolist(),
                            x=["Pred 0", "Pred 1"],
                            y=["Real 0", "Real 1"],
                            colorscale="Reds",
                            showscale=True
                        )

                        fig_cm.update_layout(
                            title_text="Matriz de Confusión",
                            xaxis_title="Predicción",
                            yaxis_title="Real"
                        )

                        fig_cm["layout"]["yaxis"]["autorange"] = "reversed"
                        st.plotly_chart(fig_cm, use_container_width=True)

                    with c2:
                        fpr, tpr, thresholds = roc_curve(y_true, preds_proba)
                        roc_auc = auc(fpr, tpr)

                        fig_roc = go.Figure()

                        fig_roc.add_trace(go.Scatter(
                            x=fpr,
                            y=tpr,
                            mode="lines",
                            name=f"ROC AUC = {roc_auc:.4f}"
                        ))

                        fig_roc.add_trace(go.Scatter(
                            x=[0, 1],
                            y=[0, 1],
                            mode="lines",
                            name="Random",
                            line=dict(dash="dash")
                        ))

                        fig_roc.update_layout(
                            title="Curva ROC",
                            xaxis_title="FPR",
                            yaxis_title="TPR"
                        )

                        st.plotly_chart(fig_roc, use_container_width=True)

                else:
                    st.warning("No se encontró columna 'attrition'. No se puede calcular evaluación contra target.")

else:
    st.info("Sube un archivo CSV para iniciar la predicción.")