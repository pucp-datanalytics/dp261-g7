import os
import json
import pandas as pd
from locust import HttpUser, task, between
from dotenv import load_dotenv

load_dotenv("dashboard/.env")

API_KEY = os.getenv("API_KEY", "demo123")

df = pd.read_csv("dashboard/X_test_sample.csv")

if "attrition" in df.columns:
    df = df.drop(columns=["attrition"])

payload = df.iloc[0].to_dict()


class MVPUser(HttpUser):
    wait_time = between(0.5, 1.5)

    headers = {
        "x-api-key": API_KEY
    }

    @task(3)
    def predict(self):
        self.client.post(
            "/predict",
            json=payload,
            headers=self.headers
        )

    @task(1)
    def health(self):
        self.client.get("/health")