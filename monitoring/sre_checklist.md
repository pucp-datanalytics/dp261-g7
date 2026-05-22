# SRE Checklist - Sprint 6

## Objetivo

Validar que la API desplegada para el MVP sea operable, monitoreable y segura a nivel básico antes de la demo.

## Alcance

Esta revisión se enfoca en la API desplegada en AWS y en sus endpoints principales:

- `/health`
- `/version`
- `/docs`
- `/predict`

## Checklist operativo

| Validación | Estado | Comentario |
|---|---|---|
| API accesible desde internet | Pendiente / Validar | Confirmar que la URL pública responda correctamente. |
| Endpoint `/health` disponible | Pendiente / Validar | Debe retornar un estado simple como `{"status": "ok"}`. |
| Endpoint `/version` disponible | Pendiente / Validar | Debe permitir verificar la versión del modelo o artefacto cargado. |
| Swagger UI `/docs` disponible | Pendiente / Validar | Debe mostrar la documentación interactiva de FastAPI. |
| Endpoint `/predict` disponible | Pendiente / Validar | Debe aceptar un JSON con datos de empleado y retornar predicción. |
| API Key requerida | Pendiente / Validar | El endpoint `/predict` debe requerir header `X-API-Key`. |
| Modelo final cargado | Pendiente / Validar | Confirmar que la API usa el modelo final definido por el equipo. |
| Respuesta de predicción clara | Pendiente / Validar | Debe retornar probabilidad y etiqueta de riesgo. |
| Logs básicos disponibles | Pendiente / Validar | Registrar fecha, endpoint, status code, latencia y error si ocurre. |
| Manejo de errores | Pendiente / Validar | Si el input es inválido, la API debe responder con error controlado. |

## Evidencia esperada

Para considerar la API lista para demo, se debe poder validar:

1. Abrir `/docs`.
2. Ejecutar `/health`.
3. Ejecutar `/version`.
4. Ejecutar `/predict` con API Key.
5. Ver una respuesta con probabilidad y label.
6. Confirmar que no se exponen credenciales en GitHub.

## Observación

La API fue desplegada sobre una instancia AWS temporal. Dado que el laboratorio puede cerrarse después de algunas horas, se recomienda validar disponibilidad antes de la exposición.