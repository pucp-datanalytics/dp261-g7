import os
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv("dashboard/.env")

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

df = pd.read_csv("dashboard/X_test_sample.csv")

if "attrition" in df.columns:
    X = df.drop(columns=["attrition"])
else:
    X = df.copy()

payload = X.iloc[0].to_dict()

response = requests.post(
    f"{API_URL}/predict",
    json=payload,
    timeout=5
)

print("STATUS SIN API KEY:", response.status_code)
print("RESPUESTA:", response.text)