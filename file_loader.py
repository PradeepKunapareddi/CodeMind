# ============================================================
# STEP 1 — FILE LOADER
# This file reads all your code files from a folder
# and returns them as plain text with their file names
# ============================================================

import os

# These are the file types we want to read
# We skip images, videos, and other non-code files
SUPPORTED_EXTENSIONS = [".py", ".js", ".java", ".ts", ".html", ".css", ".md", ".txt", ".json"]

def load_code_files(folder_path: str) -> list[dict]:
    """
    Reads all code files from a folder.
    Returns a list of dictionaries with file name and content.

    Example output:
    [
        {"filename": "app.py", "content": "from fastapi import FastAPI..."},
        {"filename": "model.py", "content": "import sklearn..."},
    ]
    """
    documents = []

    # Walk through every file and subfolder inside the given folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:

            # Get the file extension like .py .js .java
            ext = os.path.splitext(file)[1].lower()

            # Only process supported code file types
            if ext in SUPPORTED_EXTENSIONS:
                file_path = os.path.join(root, file)

                try:
                    # Open and read the file content
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                    # Skip empty files
                    if content.strip():
                        documents.append({
                            "filename": file_path,  # full path
                            "content": content       # actual code content
                        })
                        print(f"✅ Loaded: {file_path}")

                except Exception as e:
                    print(f"❌ Could not read {file_path}: {e}")

    print(f"\n📁 Total files loaded: {len(documents)}")
    return documents


# ============================================================
# TEST — Run this file directly to test the loader
# python file_loader.py
# ============================================================
if __name__ == "__main__":
    # Change this path to your project folder
    folder = "./sample_code"
    os.makedirs(folder, exist_ok=True)

    # Create a sample test file
    with open(f"{folder}/app.py", "w") as f:
        f.write("""
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Loan Prediction API")
model = joblib.load("loan_model.pkl")

class LoanInput(BaseModel):
    Gender: int
    Married: int
    ApplicantIncome: float
    LoanAmount: float

@app.get("/")
def home():
    return {"message": "Loan Prediction API is running"}

@app.post("/predict")
def predict(data: LoanInput):
    features = np.array([[data.Gender, data.Married,
                          data.ApplicantIncome, data.LoanAmount]])
    prediction = model.predict(features)[0]
    result = "Approved" if prediction == 1 else "Rejected"
    return {"prediction": int(prediction), "result": result}
""")

    docs = load_code_files(folder)
    for doc in docs:
        print(f"\nFile: {doc['filename']}")
        print(f"Content preview: {doc['content'][:100]}...")
