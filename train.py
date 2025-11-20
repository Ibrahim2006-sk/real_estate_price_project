import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load dataset
root = Path(__file__).resolve().parent
df = pd.read_csv(root / "data" / "sample_houses.csv")

# Feature engineering
df['age'] = 2025 - df['year_built']

features = [
    'area_sqft', 'bedrooms', 'bathrooms', 'age',
    'locality', 'furnishing', 'property_type',
    'latitude', 'longitude'
]

X = df[features].copy()
y = df['price']

# Columns
num_cols = ['area_sqft', 'bedrooms', 'bathrooms', 'age', 'latitude', 'longitude']
cat_cols = ['locality', 'furnishing', 'property_type']

# Pipelines
num_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())
])

cat_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preproc = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# Model
model = Pipeline([
    ('preproc', preproc),
    ('xgb', XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        verbosity=0
    ))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)

# FIXED FOR SKLEARN 1.7 → no squared=False anymore
rmse = mean_squared_error(y_test, preds)**0.5
r2 = r2_score(y_test, preds)

print(f"RMSE: {rmse:.2f}")
print(f"R2: {r2:.3f}")

# Save model
models_dir = root / "models"
models_dir.mkdir(exist_ok=True)

joblib.dump(model, models_dir / "house_price_model.joblib")
print("Saved model to models/house_price_model.joblib")
