import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "crop_data.csv"
MODEL_PATH = ROOT / "crop_model.pkl"
METRICS_PATH = ROOT / "model_metrics.txt"

def make_demo_dataset(n_per_crop=250, seed=42):
    """Create illustrative synthetic data; replace with verified farm data for real use."""
    rng = np.random.default_rng(seed)
    crop_specs = {
        "Rice": (4.2, 1100, 27, 75),
        "Wheat": (3.4, 550, 22, 55),
        "Maize": (5.0, 750, 25, 60),
        "Cotton": (2.2, 800, 27, 55),
        "Sugarcane": (70.0, 1400, 28, 75),
    }
    records = []
    for crop, (base, ideal_rain, ideal_temp, ideal_humidity) in crop_specs.items():
        for _ in range(n_per_crop):
            rainfall = rng.uniform(300, 1800)
            temperature = rng.uniform(16, 40)
            humidity = rng.uniform(35, 95)
            nitrogen = rng.uniform(20, 150)
            phosphorus = rng.uniform(10, 100)
            potassium = rng.uniform(10, 120)

            rain_score = np.exp(-((rainfall - ideal_rain) / 650) ** 2)
            temp_penalty = abs(temperature - ideal_temp) * 0.055
            humidity_score = max(0.45, 1 - abs(humidity - ideal_humidity) / 120)
            nutrient_score = (nitrogen / 150) * 0.35 + (phosphorus / 100) * 0.20 + (potassium / 120) * 0.20
            yield_value = base * (0.55 + 0.30 * rain_score + 0.15 * humidity_score)
            yield_value *= (0.75 + nutrient_score)
            yield_value -= temp_penalty * (base / 10)
            yield_value += rng.normal(0, max(0.15, base * 0.035))
            records.append([crop, rainfall, temperature, humidity, nitrogen,
                            phosphorus, potassium, max(0.1, yield_value)])

    columns = ["Crop", "Rainfall", "Temperature", "Humidity",
               "Nitrogen", "Phosphorus", "Potassium", "Yield"]
    return pd.DataFrame(records, columns=columns)

def main():
    df = make_demo_dataset()
    df.to_csv(DATA_PATH, index=False)

    X = df.drop(columns=["Yield"])
    y = df["Yield"]

    preprocessor = ColumnTransformer(
        [("crop", OneHotEncoder(handle_unknown="ignore"), ["Crop"])],
        remainder="passthrough"
    )
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=250, max_depth=14, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    METRICS_PATH.write_text(
        f"Mean Absolute Error: {mae:.3f}\\nR2 Score: {r2:.3f}\\n",
        encoding="utf-8"
    )
    print("Training complete.")
    print(f"Rows: {len(df)}")
    print(f"MAE: {mae:.3f}")
    print(f"R2 score: {r2:.3f}")
    print(f"Saved model: {MODEL_PATH.name}")
    print("NOTE: This model uses synthetic demonstration data, not real farm observations.")

if __name__ == "__main__":
    main()
