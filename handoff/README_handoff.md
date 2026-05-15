# Handoff Sprint 6

## Objetivo

Preparar los insumos mínimos necesarios para que el Sprint 6 pueda iniciar el despliegue del MVP del modelo.

## Artefactos principales

| Artefacto | Ubicación | Uso |
|---|---|---|
| Modelo final | `models/final_model.pkl` | Modelo utilizado para predicción. |
| Dashboard | `dashboard/app.py` | Aplicación Streamlit para demo e interacción. |
| Archivo de prueba | `X_test_sample.csv` | Archivo usado para validar carga y predicción. |
| Dependencias | `requirements.txt` | Librerías necesarias para ejecutar la app. |
| Validación QA | `reports/qa_validation.md` | Evidencia de pruebas realizadas. |
| Feedback | `reports/feedback.md` | Observaciones del stakeholder/reviewer. |

## Consideraciones para Sprint 6

- Confirmar que el modelo final se carga correctamente desde `models/final_model.pkl`.
- Validar que el archivo de entrada no requiera la variable objetivo `attrition`.
- Mantener un esquema claro de columnas esperadas por el modelo.
- Asegurar que `requirements.txt` contenga todas las librerías necesarias para ejecutar Streamlit.
- Mantener mensaje de carga para la sección SHAP, ya que puede tardar algunos segundos.
- Agregar explicación de negocio para predicción y probabilidad.
- Documentar ejemplos de input/output para facilitar el despliegue del MVP.

## Estado del handoff

El dashboard se encuentra disponible para demo y las observaciones QA han sido documentadas.

El paquete queda listo como base para el Sprint 6, sujeto a correcciones menores antes del despliegue final.