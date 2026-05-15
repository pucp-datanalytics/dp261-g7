import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import streamlit.components.v1 as comps
import os
import numpy as np

# Configuración de Página
st.set_page_config(page_title="Dashboard del Modelo Final", layout="wide")

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
st.markdown("Plataforma interactiva para evaluar el rendimiento del modelo y simular predicciones con explicabilidad (SHAP).")

# ==========================================
# Carga de Recursos (Caché)
# ==========================================
@st.cache_resource
def load_model():
    # Asumiendo que se ejecuta desde la raíz dp261-g7
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'final_model.pkl')
    return joblib.load(model_path)

@st.cache_data
def load_background_data():
    data_path = os.path.join(os.path.dirname(__file__), 'X_background.csv')
    if os.path.exists(data_path):
        df_bg = pd.read_csv(data_path)
        if 'attrition' in df_bg.columns:
            df_bg = df_bg.drop(columns=['attrition'])
        return df_bg
    return None

try:
    model = load_model()
    bg_data = load_background_data()
except Exception as e:
    st.error(f"Error cargando modelo o datos de fondo: {e}")
    st.stop()

# ==========================================
# Sección de KPIs
# ==========================================
st.subheader("KPIs Principales")
col1, col2, col3 = st.columns(3)

# Podríamos extraer esto de experiments_log.csv, pero usamos valores representativos
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
        <div class='kpi-value' style='color:#2196F3; font-size: 2rem;'>Stacking Pipeline</div>
        </div>""", unsafe_allow_html=True)

st.divider()

# ==========================================
# Simulador de Predicciones
# ==========================================
st.subheader("Simulador de Predicciones")
st.markdown("Sube un archivo CSV con nuevos datos para predecir.")

uploaded = st.file_uploader("Sube un CSV", type='csv')

if uploaded:
    df = pd.read_csv(uploaded)
    st.write("**Datos cargados (Primeras 5 filas):**")
    st.dataframe(df.head(5))
    
    # Separar la variable objetivo si existe
    target_col = 'attrition'
    y_true = None
    if target_col in df.columns:
        y_true = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        X = df.copy()
        
    try:
        preds = model.predict(X)
        try:
            preds_proba = model.predict_proba(X)[:, 1]
        except:
            preds_proba = preds # Fallback si no hay proba
            
        df_result = df.copy()
        df_result['Predicción'] = preds
        df_result['Probabilidad'] = preds_proba
        
        st.success("Predicciones calculadas con éxito.")
        st.dataframe(df_result[['Predicción', 'Probabilidad']].head(20))
        
        # ==========================================
        # Visualizaciones
        # ==========================================
        st.divider()
        st.subheader("Rendimiento y Explicabilidad")
        
        tab1, tab2 = st.tabs(["Explicabilidad SHAP", "Métricas Globales (Si hay Target)"])
        
        with tab1:
            st.markdown("### Explicaciones SHAP")
            st.write("SHAP permite entender la contribución de cada variable a la decisión final del modelo.")
            
            with st.spinner("Calculando valores SHAP... Esto puede tomar unos segundos."):
                # Usamos KernelExplainer directamente sobre el pipeline completo para 
                # interpretar las variables originales (sin transformar) y evitar problemas con SMOTE.
                bg_sample = shap.sample(bg_data if bg_data is not None else X, 25)
                
                feature_names = X.columns.tolist()
                
                # Creamos una funcion wrapper que devuelve solo la proba de clase 1
                # SHAP internamente pasa numpy arrays, por lo que debemos reconvertirlo a DataFrame
                def predict_fn(x_array):
                    if not isinstance(x_array, pd.DataFrame):
                        x_df = pd.DataFrame(x_array, columns=feature_names)
                    else:
                        x_df = x_array
                    return model.predict_proba(x_df)[:, 1]
                
                explainer = shap.KernelExplainer(predict_fn, bg_sample)
                
                # Explicamos todas las filas del CSV subido (limitado a 50 para no bloquear la app)
                df_to_explain = X.head(50)
                shap_values = explainer.shap_values(df_to_explain)
                
                col_s1, col_s2 = st.columns(2)
                
                with col_s1:
                    st.markdown("**Importancia Global (Summary Plot)**")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    shap.summary_plot(shap_values, df_to_explain, show=False)
                    st.pyplot(fig)
                
                with col_s2:
                    st.markdown("**Explicación Individual (Force Plot)**")
                    max_idx = len(df_to_explain) - 1
                    idx = st.slider("Selecciona la instancia a analizar:", 0, max_idx, 0)
                    
                    sv = shap_values[idx]
                    ev = explainer.expected_value
                    # Extraer un valor escalar si expected_value es un array/lista
                    try:
                        if hasattr(ev, "__len__") and len(ev) > 0:
                            ev = ev[0]
                    except:
                        pass
                        
                    force_plot = shap.force_plot(ev, sv, df_to_explain.iloc[idx])
                    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
                    comps.html(shap_html, height=300)
        
        with tab2:
            st.markdown("### Matriz de Confusión y Curva ROC")
            st.info("Para visualizar estas métricas, el CSV subido debe contener la columna objetivo 'attrition'. Si no la tiene, no podemos comparar las predicciones con la realidad.")
            
            if y_true is not None:
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    cm = confusion_matrix(y_true, preds)
                    fig_cm, ax_cm = plt.subplots(figsize=(6,4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                    ax_cm.set_title("Matriz de Confusión")
                    ax_cm.set_ylabel("Real")
                    ax_cm.set_xlabel("Predicción")
                    st.pyplot(fig_cm)
                    
                with col_m2:
                    fpr, tpr, _ = roc_curve(y_true, preds_proba)
                    roc_auc = auc(fpr, tpr)
                    fig_roc, ax_roc = plt.subplots(figsize=(6,4))
                    ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                    ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    ax_roc.set_xlim([0.0, 1.0])
                    ax_roc.set_ylim([0.0, 1.05])
                    ax_roc.set_xlabel('False Positive Rate')
                    ax_roc.set_ylabel('True Positive Rate')
                    ax_roc.set_title('Receiver Operating Characteristic (ROC)')
                    ax_roc.legend(loc="lower right")
                    st.pyplot(fig_roc)
            else:
                st.warning("No se detectó la columna objetivo 'attrition' en el CSV subido.")
            
    except Exception as e:
        st.error(f"Error durante la predicción: {e}")
else:
    st.info("Por favor, sube un archivo CSV para generar predicciones y ver las explicaciones.")
