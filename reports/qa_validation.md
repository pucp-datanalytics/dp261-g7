# QA Validation - Sprint 5

## Objetivo

Validar de extremo a extremo los principales entregables del Sprint 5: dashboard, consistencia funcional, accesibilidad para stakeholders y preparación del handoff hacia Sprint 6.

## Dashboard validado

Link de la app:

https://dp261-g7-8jkzf4ugrefxjghvo8usy3.streamlit.app/

Archivo de prueba utilizado:

`X_test_sample.csv`

## Checklist de validación

| Criterio | Estado | Evidencia / Comentario |
|---|---|---|
| La app abre correctamente | Aprobado | El dashboard publicado en Streamlit carga correctamente. |
| Permite subir un archivo CSV | Aprobado | Se cargó el archivo `X_test_sample.csv`. |
| Muestra datos cargados | Aprobado | El dashboard muestra una vista previa de las primeras filas. |
| Genera predicciones | Aprobado | Se muestran columnas de predicción y probabilidad. |
| Presenta modelo final | Aprobado | El dashboard indica uso de Stacking Pipeline. |
| Muestra sección de explicabilidad | Aprobado con observación | La sección SHAP carga correctamente, pero puede tardar algunos segundos. |
| Muestra importancia global SHAP | Aprobado | Se visualiza un summary plot con importancia global de variables. |
| Muestra explicación individual SHAP | Aprobado | Se visualiza un force plot para una instancia seleccionada. |
| Lenguaje entendible para stakeholder | Parcial | Se recomienda agregar explicación breve de qué significa predicción 0/1 y probabilidad. |
| Uso de archivo de prueba | Aprobado con observación | El archivo contiene `attrition`, por lo que debe aclararse si se usa solo como referencia o si el dashboard lo ignora al predecir. |

## Resultado de QA

El dashboard cumple con el flujo mínimo esperado para demostración: carga de CSV, visualización de datos, predicción, probabilidad y explicabilidad mediante SHAP.

## Recomendación

Se recomienda aprobar el dashboard para demo del Sprint 5, dejando documentadas las observaciones como mejoras previas al Sprint 6.