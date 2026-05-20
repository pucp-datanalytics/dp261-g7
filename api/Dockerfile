FROM python:3.12-slim

# Evitar la escritura de archivos .pyc y habilitar salida en tiempo real
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar curl para HEALTHCHECK
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copiar e instalar las dependencias primero para optimizar la caché de Docker
COPY api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copiar el código de la API y los archivos del modelo
COPY api /app/api
COPY handoff/model /app/handoff/model

# Crear un usuario no-root por seguridad y otorgar permisos sobre la carpeta /app
RUN useradd -m appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Healthcheck de Docker utilizando curl contra /health
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando por defecto para iniciar el servicio
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
