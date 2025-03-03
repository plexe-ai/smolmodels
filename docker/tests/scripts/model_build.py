import requests


def main():
    # TODO: convert standalone test scripts to integration tests
    api_url = "http://localhost:8000/models"

    input_schema = {
        "age": "int",
        "cholesterol": "float",
        "exercise": "bool",
        "smoking": "bool",
        "blood_pressure": "float",
    }

    output_schema = {"heart_attack": "bool"}

    # Create example data
    sample_data = [
        {
            "age": 45,
            "cholesterol": 230.0,
            "exercise": True,
            "smoking": False,
            "blood_pressure": 120.0,
            "heart_attack": False,
        },
        {
            "age": 62,
            "cholesterol": 300.0,
            "exercise": False,
            "smoking": True,
            "blood_pressure": 155.0,
            "heart_attack": True,
        },
        {
            "age": 52,
            "cholesterol": 210.0,
            "exercise": True,
            "smoking": True,
            "blood_pressure": 140.0,
            "heart_attack": False,
        },
        {
            "age": 70,
            "cholesterol": 320.0,
            "exercise": False,
            "smoking": True,
            "blood_pressure": 160.0,
            "heart_attack": True,
        },
        {
            "age": 58,
            "cholesterol": 250.0,
            "exercise": True,
            "smoking": False,
            "blood_pressure": 145.0,
            "heart_attack": True,
        },
    ]

    request_body = {
        "intent": "Predict whether a patient will have a heart attack based on clinical data.",
        "input_schema": input_schema,
        "output_schema": output_schema,
        "example_data": sample_data,
        "timeout": 60,
        "max_iterations": 5,
    }

    print("Sending request to create a model...")
    response = requests.post(api_url, json=request_body)

    if response.status_code == 200:
        result = response.json()
        print("Model creation job started successfully:")
        print(f"  Model ID: {result['model_id']}")
        print(f"  Job ID: {result['job_id']}")
        print(f"  Status: {result['status']}")

        job_url = f"http://localhost:8000/jobs/{result['job_id']}"
        print(f"\nChecking job status at: {job_url}")
        job_response = requests.get(job_url)

        if job_response.status_code == 200:
            job_status = job_response.json()
            print(f"Job status: {job_status['status']}")
        else:
            print(f"Failed to get job status: {job_response.status_code}")
            print(job_response.text)
    else:
        print(f"Failed to create model: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    main()
