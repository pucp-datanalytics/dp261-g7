import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

os.makedirs('models_v2', exist_ok=True)
os.makedirs('data/interim', exist_ok=True)

# 1. Data Cleaning
print("Limpiando datos...")
df = pd.read_csv('data/raw/10-employee_performance.csv')
if 'employee_id' in df.columns:
    df.drop(columns=['employee_id'], inplace=True)

# Lógica básica de limpieza del original
for col in ['age', 'monthly_salary']:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())
for col in ['department', 'education']:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

df.to_csv('data/interim/employee_performance_clean.csv', index=False)

# 2. Pipeline
print("Creando pipeline...")
X = df.drop(columns=['attrition'])
y = df['attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X_train.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, num_cols),
    ("cat", categorical_pipeline, cat_cols)
])
joblib.dump(preprocessor, 'models_v2/preprocessing_pipeline.pkl')

# 3. Modelos
print("Entrenando baselines...")
# Usamos class_weight='balanced' o scale_pos_weight
rf = Pipeline([('preproc', preprocessor), ('clf', RandomForestClassifier(random_state=42, class_weight='balanced'))])
rf.fit(X_train, y_train)
joblib.dump(rf, 'models_v2/baseline_rf.pkl')

xgb = Pipeline([('preproc', preprocessor), ('clf', XGBClassifier(random_state=42, scale_pos_weight=y_train.value_counts()[0]/y_train.value_counts()[1]))])
xgb.fit(X_train, y_train)
joblib.dump(xgb, 'models_v2/baseline_xgb.pkl')

lgb = Pipeline([('preproc', preprocessor), ('clf', LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1))])
lgb.fit(X_train, y_train)
joblib.dump(lgb, 'models_v2/baseline_lgb.pkl')

# 4. Stacking Final
print("Entrenando stacking final...")
estimators = [
    ('rf', RandomForestClassifier(random_state=42, class_weight='balanced')),
    ('xgb', XGBClassifier(random_state=42, scale_pos_weight=y_train.value_counts()[0]/y_train.value_counts()[1])),
    ('lgb', LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1))
]
stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
final_model = Pipeline([('preproc', preprocessor), ('clf', stacking)])
final_model.fit(X_train, y_train)
joblib.dump(final_model, 'models_v2/final_model.pkl')

print("Completado.")
