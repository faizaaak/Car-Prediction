
import pandas as pd

# Ganti nama file jika berbeda
file_path = "data/vehicles.parquet"

print("Membaca dataset...")
df = pd.read_parquet(file_path)

# Kolom target yang ingin diprediksi
target = "price_eur"

# Fitur awal untuk prediksi
candidate_features = [
    "make",
    "model",
    "mileage_km",
    "first_registration_year",
    "power_hp",
    "fuel_type",
    "transmission",
    "body_type",
]

# Pastikan kolom tersedia di dataset
available_features = [
    col for col in candidate_features
    if col in df.columns
]

missing_features = [
    col for col in candidate_features
    if col not in df.columns
]

if target not in df.columns:
    raise ValueError(
        f"Kolom target '{target}' tidak ditemukan. "
        f"Kolom tersedia: {df.columns.tolist()}"
    )

if not available_features:
    raise ValueError("Tidak ada fitur prediksi yang cocok.")

# X = informasi mobil
X = df[available_features].copy()

# y = harga yang ingin diprediksi
y = df[target].copy()

print("\n=== HASIL PERSIAPAN ===")
print("Jumlah baris:", len(df))
print("Fitur yang dipakai:", available_features)
print("Fitur yang belum tersedia:", missing_features)
print("Ukuran X:", X.shape)
print("Ukuran y:", y.shape)

print("\nContoh fitur mobil:")
print(X.head())

print("\nContoh target harga:")
print(y.head())
