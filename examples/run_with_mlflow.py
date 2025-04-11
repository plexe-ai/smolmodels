"""
This example demonstrates how to build a model with MLFlow tracking enabled.
"""

import smolmodels as sm
from datetime import datetime


def main():
    # Define the model in terms of the input/output function it implements
    model = sm.Model(
        intent="Predict the probability of heart attack based on patient cardiac data",
        input_schema={
            "age": int,
            "gender": int,
            "cp": int,
            "trtbps": int,
            "chol": int,
            "fbs": int,
            "restecg": int,
            "thalachh": int,
            "exng": int,
            "oldpeak": float,
            "slp": int,
            "caa": int,
            "thall": int,
        },
        output_schema={"output": int},
    )

    # Define the MLFlow callback
    mlflow_callback = sm.callbacks.MLFlowCallback(
        tracking_uri="http://127.0.0.1:8080",
        experiment_name=f"heart-disease-prediction-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )

    # Build the model with MLFlow tracking
    model.build(
        datasets=[generate_heart_data(50)],
        provider="openai/gpt-4o",
        max_iterations=5,
        callbacks=[mlflow_callback],
    )

    # Run a prediction on the built model
    print(
        model.predict(
            {
                "age": 61,
                "gender": 1,
                "cp": 3,
                "trtbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 0,
                "thalachh": 150,
                "exng": 0,
                "oldpeak": 2.3,
                "slp": 0,
                "caa": 0,
                "thall": 1,
            }
        )
    )


def generate_heart_data(n_samples=30, random_seed=42):
    import numpy as np
    import pandas as pd

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


if __name__ == "__main__":
    main()
