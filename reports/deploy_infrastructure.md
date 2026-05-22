# Infraestructura Cloud — Sprint 6 (PB-20)
**Rol:** Cloud/DevOps Engineer  
**Autor:** Miguel López  
**Fecha:** Mayo 2026  

## Arquitectura seleccionada: EC2

Se eligió EC2 sobre Lambda por las siguientes razones:
- El modelo final (21 MB) excede los límites de Lambda
- Mayor control sobre el entorno Docker
- Compatible con AWS Academy Learner Lab
- Más simple para el equipo en esta etapa académica

## Recursos creados en AWS

| Recurso | Nombre | Detalle |
|---|---|---|
| ECR Repository | dp261-g7-api | 132456600650.dkr.ecr.us-east-1.amazonaws.com/dp261-g7-api |
| S3 Bucket | dp261-g7-models | Almacena model.pkl y pipeline.pkl |
| Security Group | dp261-g7-sg | Puertos 22 (SSH) y 8000 (API) |
| EC2 Instance | dp261-g7-api | t2.micro, ami-0c02fb55956c7d316 |
| IAM Role | LabInstanceProfile | Rol preconfigurado de AWS Academy |

## Endpoint público

http://98.90.197.248:8000

### Endpoints disponibles
- `GET /health` — Estado de la API
- `POST /predict` — Predicción de attrition
- `GET /version` — Versión del modelo

## Pasos de despliegue realizados

1. Construcción de imagen Docker localmente
2. Creación de repositorio ECR
3. Push de imagen a ECR
4. Creación de bucket S3 y subida de modelos
5. Configuración de Security Group
6. Lanzamiento de instancia EC2 con user-data
7. Verificación del health check

## Verificación

```bash
curl http://98.90.197.248:8000/health
# Respuesta: {"status":"ok"}
```

## Notas importantes

- Las credenciales de AWS Academy expiran cada 4 horas
- Al reiniciar el Lab, actualizar credenciales y reiniciar el contenedor en EC2
- Presupuesto usado: $0 de $50 disponibles
- Región: us-east-1