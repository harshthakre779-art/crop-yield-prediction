from pathlib import Path
import pickle
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "crop_model.pkl"
DATA_PATH = ROOT / "crop_data.csv"
METRICS_PATH = ROOT / "model_metrics.txt"

st.set_page_config(page_title="Crop Yield Prediction", page_icon="🌱", layout="wide")

st.title("🌾 Crop Yield Prediction & Agricultural Decision Support")
st.caption("A machine-learning demonstration built with Python, Scikit-learn and Streamlit.")

if not MODEL_PATH.exists():
    st.error("Trained model not found. Open the VS Code terminal and run: python train_model.py")
    st.stop()

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()

with st.sidebar:
    st.header("About the project")
    st.write("The app estimates crop yield from crop type, weather conditions and N-P-K soil nutrient inputs.")
    st.info("Demo limitation: the included training data is synthetic. Do not use these estimates for real farming decisions.")
    if METRICS_PATH.exists():
        st.subheader("Model evaluation")
        st.code(METRICS_PATH.read_text(encoding="utf-8"))

tab_predict, tab_data, tab_about = st.tabs(["🌱 Yield prediction", "📊 Dataset explorer", "ℹ️ About"])

with tab_predict:
    st.subheader("Enter field conditions")
    st.write("Use the same units shown beside each input. Nutrient values are illustrative input levels, not laboratory recommendations.")
    c1, c2, c3 = st.columns(3)
    with c1:
        crop = st.selectbox("Crop type", ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane"])
        rainfall = st.number_input("Annual rainfall (mm)", min_value=0.0, max_value=5000.0, value=800.0, step=25.0)
        temperature = st.number_input("Average temperature (°C)", min_value=0.0, max_value=55.0, value=27.0, step=0.5)
    with c2:
        humidity = st.number_input("Average humidity (%)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
        nitrogen = st.number_input("Nitrogen (N) input level", min_value=0.0, max_value=300.0, value=50.0, step=5.0)
    with c3:
        phosphorus = st.number_input("Phosphorus (P) input level", min_value=0.0, max_value=300.0, value=40.0, step=5.0)
        potassium = st.number_input("Potassium (K) input level", min_value=0.0, max_value=300.0, value=40.0, step=5.0)

    if st.button("Predict crop yield", type="primary", use_container_width=True):
        row = pd.DataFrame([{
            "Crop": crop, "Rainfall": rainfall, "Temperature": temperature,
            "Humidity": humidity, "Nitrogen": nitrogen,
            "Phosphorus": phosphorus, "Potassium": potassium
        }])
        value = float(model.predict(row)[0])
        st.success("Prediction generated successfully.")
        st.metric("Estimated crop yield", f"{value:,.2f} tonnes/hectare")
        st.caption("This is a model estimate from synthetic demonstration data, not a validated forecast.")
        st.session_state["last_prediction"] = {
            "Crop": crop, "Rainfall (mm)": rainfall, "Temperature (°C)": temperature,
            "Humidity (%)": humidity, "Nitrogen": nitrogen, "Phosphorus": phosphorus,
            "Potassium": potassium, "Predicted yield (tonnes/hectare)": round(value, 2)
        }

    if "last_prediction" in st.session_state:
        st.subheader("Latest prediction details")
        st.dataframe(pd.DataFrame([st.session_state["last_prediction"]]), use_container_width=True)
        csv = pd.DataFrame([st.session_state["last_prediction"]]).to_csv(index=False).encode("utf-8")
        st.download_button("Download prediction as CSV", csv, "crop_yield_prediction.csv", "text/csv")

with tab_data:
    st.subheader("Dataset overview")
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        a, b, c = st.columns(3)
        a.metric("Records", f"{len(df):,}")
        b.metric("Crop types", df["Crop"].nunique())
        c.metric("Features", len(df.columns) - 1)
        st.dataframe(df.head(20), use_container_width=True)
        st.subheader("Average yield by crop")
        chart = df.groupby("Crop")["Yield"].mean().sort_values(ascending=False)
        st.bar_chart(chart)
    else:
        st.warning("Dataset not found. Run train_model.py first.")

with tab_about:
    st.subheader("Project modules")
    st.markdown("""
    1. **Data generation / collection** – prepares the sample crop dataset.
    2. **Preprocessing** – one-hot encodes crop type.
    3. **Model training** – trains a Random Forest Regressor.
    4. **Prediction** – estimates yield from user-entered conditions.
    5. **Evaluation** – reports MAE and R² on a held-out test split.
    6. **Visualization** – displays dataset records and average yield by crop.
    """)
    st.warning("For a genuine academic or agricultural deployment, replace the synthetic dataset with verified, location-specific historical yield and weather/soil data, then retrain and validate the model.")
    st.write("Target unit: tonnes per hectare. The app's sample data is illustrative only.")
