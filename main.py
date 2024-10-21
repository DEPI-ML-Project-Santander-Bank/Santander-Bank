from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import io

# Load pre-trained models, scaler, and PCA
models = joblib.load('lightgbm_model.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')

# Initialize FastAPI app
app = FastAPI()

# Endpoint 1: Health check to ensure the API is running
@app.get("/")
def read_root():
    return {"message": "API is running"}

# Function to make predictions on the test data
def make_test_predictions(test_df):
    # Clean the test data (note that test data doesn't have 'target' column)
    test_cleaned = test_df.drop(columns=['ID_code']).copy()

    # Handle any missing values in test data similarly to training data
    test_cleaned.fillna(test_cleaned.mean(), inplace=True)

    # Apply scaling to the test data
    test_scaled = scaler.transform(test_cleaned)

    # Apply PCA to the scaled test data
    test_pca = pca.transform(test_scaled)

    # Predict probabilities using trained models and save results
    predictions = models.predict_proba(test_pca)[:, 1]

    # Add predictions to the DataFrame
    test_df['target'] = predictions
    return test_df[['ID_code', 'target']]

# Endpoint 2: POST request to accept CSV and return predictions in CSV format
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded CSV file into a DataFrame
    contents = await file.read()
    test_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

    # Make predictions
    predictions_df = make_test_predictions(test_df)

    # Convert DataFrame back to CSV format
    output = io.StringIO()
    predictions_df.to_csv(output, index=False)

    # Set the file pointer to the beginning of the stream
    output.seek(0)

    # Create a StreamingResponse to return the CSV file
    response = StreamingResponse(io.BytesIO(output.getvalue().encode('utf-8')), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=predictions.csv"
    
    return response

# To run the server, use the following command in your terminal:
# uvicorn app:app --reload