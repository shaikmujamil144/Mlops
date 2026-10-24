"""Task 5: Call the local FastAPI /predict endpoint from Python."""
import requests

BASE_URL = "http://127.0.0.1:8000"

def main():
    passenger = {
        "Pclass": 3, "Sex": "male", "Age": 22, "SibSp": 1,
        "Parch": 0, "Fare": 7.25, "Embarked": "S"
    }
    response = requests.post(f"{BASE_URL}/predict", json=passenger, timeout=10)
    print("HTTP status:", response.status_code)
    print("Response:", response.text)
    response.raise_for_status()

if __name__ == "__main__":
    main()
