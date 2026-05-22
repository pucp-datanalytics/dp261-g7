# Runbook Operativo - Sprint 6

## Objetivo

Definir acciones básicas ante incidentes operativos de la API desplegada en AWS.

## Escenario 1: La API no responde

### Síntoma

La URL pública no carga o el endpoint `/health` no responde.

### Validación

1. Verificar si la instancia AWS sigue activa.
2. Probar la URL base de la API.
3. Probar `/health`.
4. Revisar si el laboratorio AWS sigue encendido.

### Acción recomendada

- Reiniciar la instancia si el laboratorio sigue activo.
- Volver a ejecutar la API con Uvicorn si el proceso se detuvo.
- Si cambió la IP pública, actualizar la URL usada por el dashboard o documentación.

---

## Escenario 2: `/health` falla

### Síntoma

El endpoint `/health` no retorna estado `ok`.

### Validación

1. Confirmar que el servidor está encendido.
2. Revisar si el puerto 8000 está disponible.
3. Revisar logs de la API.

### Acción recomendada

- Reiniciar el servicio de la API.
- Validar que FastAPI esté corriendo correctamente.
- Confirmar que la ruta `/health` esté definida en `api/main.py`.

---

## Escenario 3: `/predict` retorna error

### Síntoma

La API no genera predicción o retorna error 4xx/5xx.

### Validación

1. Confirmar que se envía el header `X-API-Key`.
2. Confirmar que el `Content-Type` sea `application/json`.
3. Validar que el JSON tenga las columnas esperadas.
4. Revisar si el modelo final está cargado correctamente.

### Acción recomendada

- Corregir el formato del request.
- Confirmar que el modelo exista en la ruta esperada.
- Revisar logs para identificar si el error es por input, API Key o modelo.

---

## Escenario 4: Latencia alta

### Síntoma

La API responde, pero demora demasiado.

### Validación

1. Medir tiempo de respuesta de `/predict`.
2. Revisar si el cálculo del modelo o SHAP está demorando.
3. Revisar consumo de CPU/memoria de la instancia.

### Acción recomendada

- Mantener SHAP fuera del flujo crítico de predicción si afecta la respuesta.
- Revisar tamaño del input.
- Considerar una instancia con mayores recursos si el MVP escala.

---

## Escenario 5: Dashboard no conecta con API

### Síntoma

El dashboard carga, pero no obtiene predicciones.

### Validación

1. Probar la API directamente en `/docs`.
2. Confirmar que la URL de la API esté actualizada.
3. Confirmar que la IP pública no haya cambiado.
4. Confirmar que el puerto 8000 esté abierto.

### Acción recomendada

- Actualizar la URL base en el dashboard.
- Confirmar que la instancia AWS esté activa.
- Validar que la API Key esté correctamente configurada.

---

## Contacto y escalamiento

Si el incidente no se resuelve:

1. Avisar al Cloud / DevOps Engineer.
2. Avisar al API Developer.
3. Documentar el error, endpoint afectado y hora del incidente.
4. Si aplica, usar la versión local como respaldo para la demo.