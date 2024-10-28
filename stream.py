import streamlit as st
import pandas as pd
import joblib
import io

# Set page configuration
st.set_page_config(
    page_title="Santander Customer Transaction Prediction",
    page_icon=":bar_chart:",  # You can use an emoji or a URL to an image
    layout="centered"
)

# Load pre-trained models, scaler, and PCA
models = joblib.load('lightgbm_model.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')

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

# Streamlit app
st.title("Santander Customer Transaction Prediction")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the uploaded CSV file into a DataFrame
    test_df = pd.read_csv(uploaded_file)

    # Make predictions
    predictions_df = make_test_predictions(test_df)

    # Display predictions
    st.write(predictions_df)

    # Display charts
    st.subheader("Prediction Threshold Distribution")
    threshold_counts = predictions_df['target'].apply(lambda x: '1' if x >= 0.5 else '0').value_counts()
    st.bar_chart(threshold_counts)

    st.subheader("Prediction Distribution in Sections")
    section_counts = pd.cut(predictions_df['target'], bins=[0, 0.25, 0.5, 0.75, 1], labels=['0-0.25', '0.25-0.5', '0.5-0.75', '0.75-1']).value_counts()
    st.bar_chart(section_counts,color='orange')

    # Convert DataFrame back to CSV format
    output = io.StringIO()
    predictions_df.to_csv(output, index=False)
    output.seek(0)

    # Provide download link for the predictions CSV file
    st.download_button(
        label="Download Predictions",
        data=output.getvalue(),
        file_name="predictions.csv",
        mime="text/csv"
    )