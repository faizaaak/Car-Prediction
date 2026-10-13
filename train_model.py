
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Membaca dataset
print("Membaca dataset...")
df = pd.read_parquet("data/vehicles.parquet")

target = "price_eur"

features = [
    "make",
    "model",
    "mileage_km",
    "first_registration_year",
    "power_hp",
    "fuel_type",
    "transmission",
    "body_type",
]

X = df[features].copy()
y = df[target].copy()

# 2. Menghapus baris yang target harganya tidak valid
valid = y.notna() & (y > 0)
X = X.loc[valid]
y = y.loc[valid]

# 3. Memisahkan fitur kategori dan numerik
categorical_features = [
    "make",
    "model",
    "fuel_type",
    "transmission",
    "body_type",
]

numeric_features = [
    "mileage_km",
    "first_registration_year",
    "power_hp",
]

# 4. Menyiapkan preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

# 5. Membagi data training dan testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print("Data training:", len(X_train))
print("Data testing:", len(X_test))

# 6. Membuat model AI
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )),
])

# 7. Training
print("\nAI mulai belajar... tunggu sampai selesai.")
model.fit(X_train, y_train)
print("Training selesai!")

# 8. Menguji hasil prediksi
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\n=== HASIL EVALUASI MODEL ===")
print(f"MAE  : €{mae:,.2f}")
print(f"RMSE : €{rmse:,.2f}")
print(f"R²   : {r2:.4f}")

# 9. Menyimpan model
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/car_price_model.joblib")

print("\nModel berhasil disimpan!")
print("Lokasi: models/car_price_model.joblib")
