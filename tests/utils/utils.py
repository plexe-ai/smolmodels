# tests/utils.py
import numpy as np
import pandas as pd


def generate_heart_data(n_samples=200, random_seed=42):
    """Generate synthetic heart disease data for testing.

    The data follows the structure:
    - age: int (25-80)
    - gender: int (0=female, 1=male)
    - cp: int (chest pain type, 0-3)
    - trtbps: int (resting blood pressure, 90-200)
    - chol: int (cholesterol, 120-400)
    - fbs: int (fasting blood sugar > 120 mg/dl, 0-1)
    - restecg: int (resting ECG results, 0-2)
    - thalachh: int (maximum heart rate achieved, 70-220)
    - exng: int (exercise induced angina, 0-1)
    - oldpeak: float (ST depression induced by exercise, 0-6.0)
    - slp: int (slope of peak exercise ST segment, 0-2)
    - caa: int (number of major vessels, 0-4)
    - thall: int (thalassemia, 0-3)
    - output: int (presence of heart disease, 0-1)
    """
    np.random.seed(random_seed)

    # Generate features
    data = {
        "age": np.random.randint(25, 80, n_samples),
        "gender": np.random.randint(0, 2, n_samples),
        "cp": np.random.randint(0, 4, n_samples),
        "trtbps": np.random.randint(90, 200, n_samples),
        "chol": np.random.randint(120, 400, n_samples),
        "fbs": np.random.randint(0, 2, n_samples),
        "restecg": np.random.randint(0, 3, n_samples),
        "thalachh": np.random.randint(70, 220, n_samples),
        "exng": np.random.randint(0, 2, n_samples),
        "oldpeak": np.round(np.random.uniform(0, 6, n_samples), 1),
        "slp": np.random.randint(0, 3, n_samples),
        "caa": np.random.randint(0, 5, n_samples),
        "thall": np.random.randint(0, 4, n_samples),
    }

    # Generate target based on risk factors
    risk_factors = (
        (data["age"] > 60).astype(int) * 2  # Age over 60 is high risk
        + data["gender"]  # Being male slightly increases risk
        + (data["cp"] > 1).astype(int) * 2  # Higher chest pain types increase risk
        + (data["trtbps"] > 140).astype(int)  # High blood pressure
        + (data["chol"] > 250).astype(int)  # High cholesterol
        + data["fbs"]  # High fasting blood sugar
        + (data["thalachh"] < 120).astype(int) * 2  # Low max heart rate
        + data["exng"] * 2  # Exercise-induced angina
        + (data["oldpeak"] > 2).astype(int) * 2  # High ST depression
        + data["caa"]  # Number of major vessels
    )

    # Convert risk factors to binary output (threshold chosen to get roughly balanced classes)
    data["output"] = (risk_factors > 8).astype(int)

    return pd.DataFrame(data)
