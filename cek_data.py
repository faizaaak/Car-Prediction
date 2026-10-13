
import pandas as pd

# Ganti nama file sesuai file Parquet milikmu
file_path = "data/vehicles.parquet"

# Membaca dataset
df = pd.read_parquet(file_path)

# Informasi dasar
print("=== UKURAN DATASET ===")
print(f"Jumlah baris: {df.shape[0]:,}")
print(f"Jumlah kolom: {df.shape[1]}")

print("\n=== NAMA KOLOM ===")
print(df.columns.tolist())

print("\n=== 5 BARIS PERTAMA ===")
print(df.head())

print("\n=== TIPE DATA ===")
print(df.dtypes)

print("\n=== MISSING VALUES ===")
print(df.isna().sum().sort_values(ascending=False))


print("\n=== RINGKASAN KUALITAS DATA ===")

# Jumlah baris duplikat
print("Baris duplikat:", df.duplicated().sum())

# Persentase data kosong
print("\nPersentase data kosong (%):")
print((df.isna().mean() * 100).round(2).sort_values(ascending=False))

# Statistik kolom numerik
print("\nStatistik kolom numerik:")
print(df.describe())

# Cek kolom target harga
if "price_eur" in df.columns:
    print("\n=== PEMERIKSAAN HARGA MOBIL ===")
    print("Harga kosong:", df["price_eur"].isna().sum())
    print("Harga nol atau negatif:", (df["price_eur"] <= 0).sum())
    print("Harga minimum:", df["price_eur"].min())
    print("Harga maksimum:", df["price_eur"].max())
    print("Median harga:", df["price_eur"].median())
else:
    print("\nKolom 'price_eur' tidak ditemukan.")
    print("Kita perlu menentukan nama kolom harga yang benar.")


print("\n=== VALIDASI DATA MOBIL ===")

if "price_eur" in df.columns:
    print("Jumlah harga <= 0:", (df["price_eur"] <= 0).sum())
    print("Jumlah harga di bawah €500:", (df["price_eur"] < 500).sum())
    print("Jumlah harga di atas €200.000:", (df["price_eur"] > 200000).sum())
    print("\nContoh 10 harga terendah:")
    print(df["price_eur"].nsmallest(10).to_list())

    print("\nContoh 10 harga tertinggi:")
    print(df["price_eur"].nlargest(10).to_list())
else:
    print("Kolom price_eur tidak ditemukan.")
    print("Nama kolom yang tersedia:")
    print(df.columns.tolist())

print("\n=== DISTRIBUSI HARGA ===")

print(df["price_eur"].describe(
    percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
))

print("\n=== CONTOH MOBIL DI ATAS €200.000 ===")

cols_to_show = [
    col for col in [
        "make", "model", "version", "price_eur",
        "mileage_km", "first_registration_year", "power_hp"
    ]
    if col in df.columns
]

print(
    df.loc[df["price_eur"] > 200000, cols_to_show]
    .sort_values("price_eur", ascending=False)
    .head(15)
    .to_string(index=False)
)

print("\n=== TIPE DATA TARGET ===")
print(df["price_eur"].dtype)
