# Monitoring Plan - Sprint 6

## Objetivo

Definir el plan básico de monitoreo para la API del MVP desplegada en AWS.

## Componentes monitoreados

| Componente | Qué se monitorea | Motivo |
|---|---|---|
| API FastAPI | Disponibilidad, errores y latencia | Confirmar que el servicio responde correctamente. |
| Endpoint `/health` | Estado operativo | Validar si la API está viva. |
| Endpoint `/version` | Versión del modelo/API | Asegurar trazabilidad del artefacto desplegado. |
| Endpoint `/predict` | Predicciones, errores y tiempos de respuesta | Validar funcionamiento del núcleo del MVP. |
| Instancia AWS | Estado, CPU, memoria y disponibilidad | Evitar caídas durante demo o uso. |
| Dashboard | Conexión con API | Confirmar que el usuario final puede consumir predicciones. |

## Métricas recomendadas

| Métrica | Descripción | Umbral sugerido |
|---|---|---|
| Availability | Porcentaje de tiempo que la API responde | Mayor a 95% en demo |
| Latency p95 | Tiempo de respuesta del 95% de requests | Menor a 2 segundos |
| Error rate 5xx | Porcentaje de errores del servidor | Menor a 5% |
| Error rate 4xx | Errores por request inválido o API Key incorrecta | Monitorear, no necesariamente incidente |
| Throughput | Cantidad de requests por minuto | Referencial para uso del MVP |
| CPU usage | Uso de CPU de la instancia AWS | Alerta si supera 80% sostenido |
| Memory usage | Uso de memoria | Alerta si supera 80% sostenido |

## Logs recomendados

La API debería registrar logs estructurados con los siguientes campos:

```json
{
  "timestamp": "2026-05-XXT00:00:00",
  "endpoint": "/predict",
  "method": "POST",
  "status_code": 200,
  "latency_ms": 120,
  "model_version": "final_model",
  "prediction_label": 0,
  "prediction_probability": 0.18,
  "error": null
}