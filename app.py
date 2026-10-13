
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# =========================
# 1. KONFIGURASI APLIKASI
# =========================
st.set_page_config(
    page_title="AI Car Price Predictor",
    page_icon="🚗",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "car_price_model.joblib"
DATA_PATH = BASE_DIR / "data" / "vehicles.parquet"

FEATURES = [
    "make",
    "model",
    "mileage_km",
    "first_registration_year",
    "power_hp",
    "fuel_type",
    "transmission",
    "body_type",
]


# =========================
# 2. MEMUAT MODEL DAN DATA
# =========================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_parquet(DATA_PATH)


try:
    model = load_model()
    df = load_data()
except FileNotFoundError as e:
    st.error(f"File tidak ditemukan: {e.filename}")
    st.info(
        "Pastikan model berada di folder models "
        "dan dataset berada di folder data."
    )
    st.stop()
except Exception as e:
    st.error(f"Gagal memuat model atau dataset: {e}")
    st.stop()


missing = [col for col in FEATURES if col not in df.columns]

if missing:
    st.error(f"Kolom dataset tidak ditemukan: {missing}")
    st.stop()


# =========================
# 3. FUNGSI BANTU
# =========================
def get_options(series):
    values = series.dropna().unique().tolist()
    return sorted(values, key=lambda x: str(x).lower())


def get_number_bounds(column, fallback_min, fallback_max):
    values = pd.to_numeric(df[column], errors="coerce").dropna()

    if values.empty:
        return fallback_min, fallback_max

    low = int(values.min())
    high = int(values.max())

    if low == high:
        high = low + 1

    return low, high


def euro(value):
    return f"€{value:,.0f}"


# =========================
# 4. SIDEBAR
# =========================
with st.sidebar:
    st.title("🚗 Car Predictor")
    st.caption("Machine Learning Project")
    st.divider()

    st.write("**Model:** Random Forest Regressor")
    st.write(f"**Dataset:** {len(df):,} listing")
    st.write("**Target:** Harga mobil dalam EUR")

    st.divider()
    st.caption(
        "Prediksi merupakan estimasi berdasarkan pola "
        "dataset, bukan harga jual yang dijamin."
    )


# =========================
# 5. HEADER
# =========================
st.title("🚘 AI Car Price Predictor")
st.write(
    "Masukkan spesifikasi mobil untuk memperkirakan "
    "harga mobil bekas menggunakan Machine Learning."
)

st.divider()

# =========================
# 6. FORM INPUT MOBIL
# =========================
st.subheader("1. Spesifikasi mobil")

with st.form("car_prediction_form"):
    col1, col2 = st.columns(2)

    makes = get_options(df["make"])

    with col1:
        make = st.selectbox(
            "Merek mobil",
            makes,
        )

        make_df = df[df["make"] == make]

        models = get_options(make_df["model"])

        selected_model = st.selectbox(
            "Model mobil",
            models,
        )

        year_min, year_max = get_number_bounds(
            "first_registration_year", 1990, 2026
        )

        year_default = int(
            pd.to_numeric(
                make_df["first_registration_year"],
                errors="coerce",
            ).median()
        ) if make_df["first_registration_year"].notna().any() else year_max

        year = st.number_input(
            "Tahun registrasi pertama",
            min_value=year_min,
            max_value=year_max,
            value=max(year_min, min(year_default, year_max)),
            step=1,
        )

        mileage_min, mileage_max = get_number_bounds(
            "mileage_km", 0, 500000
        )

        mileage = st.number_input(
            "Jarak tempuh (km)",
            min_value=0,
            max_value=max(mileage_max, 1_000_000),
            value=60000,
            step=5000,
        )

    with col2:
        hp_min, hp_max = get_number_bounds(
            "power_hp", 1, 1000
        )

        power = st.number_input(
            "Tenaga mesin (HP)",
            min_value=1,
            max_value=max(hp_max, 2000),
            value=min(150, max(hp_max, 2000)),
            step=10,
        )

        fuel = st.selectbox(
            "Jenis bahan bakar",
            get_options(df["fuel_type"]),
        )

        transmission = st.selectbox(
            "Transmisi",
            get_options(df["transmission"]),
        )

        body_type = st.selectbox(
            "Tipe bodi",
            get_options(df["body_type"]),
        )

    submitted = st.form_submit_button(
        "🔍 Prediksi Harga Mobil",
        use_container_width=True,
        type="primary",
    )


# =========================
# 7. PREDIKSI HARGA
# =========================
if submitted:
    input_data = pd.DataFrame(
        [{
            "make": make,
            "model": selected_model,
            "mileage_km": mileage,
            "first_registration_year": year,
            "power_hp": power,
            "fuel_type": fuel,
            "transmission": transmission,
            "body_type": body_type,
        }],
        columns=FEATURES,
    )

    try:
        prediction = float(model.predict(input_data)[0])

        st.divider()
        st.subheader("2. Hasil prediksi")

        if prediction <= 0:
            st.warning(
                "Model menghasilkan harga yang tidak masuk akal. "
                "Coba periksa spesifikasi mobil."
            )
        else:
            st.success("Prediksi berhasil dibuat!")

            st.metric(
                label="Estimasi harga mobil",
                value=euro(prediction),
            )

            st.caption(
                "Ini estimasi model, bukan penawaran resmi "
                "atau jaminan harga pasar."
            )

            st.subheader("Ringkasan spesifikasi")

            result_data = {
                "Spesifikasi": [
                    "Merek",
                    "Model",
                    "Tahun registrasi",
                    "Jarak tempuh",
                    "Tenaga mesin",
                    "Bahan bakar",
                    "Transmisi",
                    "Tipe bodi",
                ],
                "Nilai": [
                    make,
                    selected_model,
                    str(year),
                    f"{mileage:,} km",
                    f"{power} HP",
                    fuel,
                    transmission,
                    body_type,
                ],
            }

            st.table(pd.DataFrame(result_data))

    except Exception as e:
        st.error(f"Gagal memprediksi harga: {e}")


# =========================
# 8. FOOTER
# =========================
st.divider()
st.caption(
    "AI Car Price Predictor | Python • Pandas • "
    "Scikit-learn • Streamlit"
)
