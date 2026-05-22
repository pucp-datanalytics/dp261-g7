import os
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv("dashboard/.env")

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("API_KEY", "demo123")

# 1. Smoke test /health
health = requests.get(f"{API_URL}/health", timeout=5)
print("HEALTH STATUS:", health.status_code)
print("HEALTH RESPONSE:", health.json())

# 2. Test /predict usando una fila del CSV
df = pd.read_csv("dashboard/X_test_sample.csv")

if "attrition" in df.columns:
    X = df.drop(columns=["attrition"])
else:
    X = df.copy()

payload = X.iloc[0].to_dict()

response = requests.post(
    f"{API_URL}/predict",
    json=payload,
    headers={"x-api-key": API_KEY},
    timeout=5
)

print("PREDICT STATUS:", response.status_code)
print("PREDICT RESPONSE:", response.json())