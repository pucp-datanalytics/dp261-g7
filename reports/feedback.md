# Stakeholder Feedback - Sprint 5

## Objetivo

Registrar feedback funcional y de negocio sobre el dashboard y los entregables del Sprint 5, considerando la perspectiva de un stakeholder no técnico.

## Feedback recibido

| Aspecto evaluado | Comentario | Acción recomendada |
|---|---|---|
| Claridad del dashboard | El dashboard permite cargar datos y obtener predicciones de forma directa. | Mantener flujo simple para la demo. |
| Interpretación de predicciones | No queda completamente claro qué representa la predicción 0/1. | Agregar leyenda: 0 = sin riesgo, 1 = riesgo de attrition. |
| Probabilidad | La probabilidad se muestra, pero podría explicarse mejor. | Agregar texto: “Probabilidad estimada de pertenecer a la clase de riesgo”. |
| Explicabilidad SHAP | SHAP carga correctamente y permite visualizar importancia global e individual. | Mantener la sección, pero conservar mensaje de carga porque puede tardar. |
| Archivo de entrada | El CSV de prueba contiene la columna `attrition`. | Para producción, definir input sin target o aclarar que se ignora al predecir. |
| Preparación para Sprint 6 | Se requiere dejar claro qué modelo y dependencias usar. | Preparar handoff con modelo, requirements y contratos de entrada/salida. |

## Conclusión del feedback

El dashboard es útil como prototipo de demostración para stakeholders, ya que permite cargar datos, visualizar predicciones y revisar explicabilidad mediante SHAP.

Para mejorar su uso en despliegue, se recomienda reforzar la explicación de resultados, documentar el esquema de entrada y mantener control sobre las columnas requeridas por el modelo.