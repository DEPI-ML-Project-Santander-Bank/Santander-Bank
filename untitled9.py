import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, accuracy_score, classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv(r"E:\my_work\dpi\santander-customer-transaction-prediction\train.csv")  # Adjust this path accordingly

# Clean and preprocess the data
def clean_data(df, corr_threshold=0.9):
    # Drop ID column (not useful for prediction)
    df = df.drop(columns=['ID_code'])

    # Check for null values and fill if necessary
    df.fillna(df.mean(), inplace=True)

    # Drop highly correlated features (correlation threshold > 0.9)
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
    
    df = df.drop(columns=to_drop)

    return df

# Train-test split
def split_data(df):
    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

# Apply PCA for dimensionality reduction
def apply_pca(X_train, X_test, n_components=0.95):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca

# Model training function
def train_models(X_train, y_train):
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'),
        'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=10, random_state=42, class_weight='balanced'),
        'LogisticRegression': LogisticRegression(C=1.0, solver='lbfgs', random_state=42, class_weight='balanced')
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models

# Evaluate the models
def evaluate_models(models, X_train, y_train, X_test, y_test):
    for name, model in models.items():
        y_train_pred = model.predict(X_train)
        y_train_pred_prob = model.predict_proba(X_train)[:, 1]
        y_test_pred = model.predict(X_test)
        y_test_pred_prob = model.predict_proba(X_test)[:, 1]
        
        print(f"{name} Training Metrics:")
        print("Accuracy: ", accuracy_score(y_train, y_train_pred))
        print("F1 Score: ", f1_score(y_train, y_train_pred))
        print("ROC-AUC: ", roc_auc_score(y_train, y_train_pred_prob))
        print(classification_report(y_train, y_train_pred))
        print("\n")
        
        print(f"{name} Test Metrics:")
        print("Accuracy: ", accuracy_score(y_test, y_test_pred))
        print("F1 Score: ", f1_score(y_test, y_test_pred))
        print("ROC-AUC: ", roc_auc_score(y_test, y_test_pred_prob))
        print(classification_report(y_test, y_test_pred))
        print("\n")

# Main function
def main():
    # Clean data
    df = clean_data(data)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Scale the data for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply PCA for dimensionality reduction
    X_train_pca, X_test_pca = apply_pca(X_train_scaled, X_test_scaled)
    
    # Train models
    models = train_models(X_train_pca, y_train)
    
    # Evaluate models
    evaluate_models(models, X_train_pca, y_train, X_test_pca, y_test)

# Run the main function
if __name__ == "__main__":
    main()
