import requests


def main():
    model_id = "model-id"

    api_url = f"http://localhost:8000/predictions/models/{model_id}"

    # Example "new patient" data
    new_patient = {"age": 65, "cholesterol": 280.0, "exercise": False, "smoking": True, "blood_pressure": 150.0}

    request_body = {"data": new_patient}

    print(f"Sending prediction request for model {model_id}...")
    response = requests.post(api_url, json=request_body)

    if response.status_code == 200:
        result = response.json()
        print("Prediction result:")
        print(f"  Heart attack: {result['prediction']['heart_attack']}")
    else:
        print(f"Failed to get prediction: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    main()
