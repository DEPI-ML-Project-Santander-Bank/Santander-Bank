import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, accuracy_score, classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
data = pd.read_csv(r"E:\my_work\dpi\santander-customer-transaction-prediction\train.csv")  # Adjust this path accordingly
test = pd.read_csv(r"E:\my_work\dpi\santander-customer-transaction-prediction\test.csv")  # Adjust this path accordingly

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

# Apply PCA for dimensionality reduction
def apply_pca(X_train, X_test, n_components=0.95):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca, pca

# Model training function using StratifiedKFold
def train_and_evaluate_models(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = {
        'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(**{
     'learning_rate': 0.04,
     'num_leaves': 31,
     'max_bin': 1023,
     'min_child_samples': 1000,
     'reg_alpha': 0.1,
     'reg_lambda': 0.2,
     'feature_fraction': 1.0,
     'bagging_freq': 1,
     'bagging_fraction': 0.85,
     'objective': 'binary',
     'n_jobs': -1,
     'n_estimators':200,})
    }

    fold_metrics = {name: {'accuracy': [], 'f1_score': [], 'roc_auc': []} for name in models.keys()}

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Apply PCA for dimensionality reduction
        X_train_pca, X_test_pca, pca = apply_pca(X_train_scaled, X_test_scaled)
        
        for name, model in models.items():
            # Train the model
            model.fit(X_train_pca, y_train)
            
            # Make predictions
            y_test_pred = model.predict(X_test_pca)
            y_test_pred_prob = model.predict_proba(X_test_pca)[:, 1]
            
            # Evaluation metrics
            accuracy = accuracy_score(y_test, y_test_pred)
            f1 = f1_score(y_test, y_test_pred)
            roc_auc = roc_auc_score(y_test, y_test_pred_prob)
            
            # Store metrics for each fold
            fold_metrics[name]['accuracy'].append(accuracy)
            fold_metrics[name]['f1_score'].append(f1)
            fold_metrics[name]['roc_auc'].append(roc_auc)

    # Print overall metrics for each model
    for name, metrics in fold_metrics.items():
        print(f"\n{name} Cross-Validation Metrics:")
        print(f"Mean Accuracy: {np.mean(metrics['accuracy']):.2f}")
        print(f"Mean F1 Score: {np.mean(metrics['f1_score']):.2f}")
        print(f"Mean ROC-AUC: {np.mean(metrics['roc_auc']):.2f}")

    # Return trained models, scaler, and pca from the last fold for predictions
    return models, scaler, pca

# Function to make predictions on the test set and save the output
def make_test_predictions(models, scaler, pca):
    # Clean the test data (note that test data doesn't have 'target' column)
    test_cleaned = test.drop(columns=['ID_code']).copy()

    # Handle any missing values in test data similarly to training data
    test_cleaned.fillna(test_cleaned.mean(), inplace=True)

    # Apply scaling to the test data
    test_scaled = scaler.transform(test_cleaned)

    # Apply PCA to the scaled test data
    test_pca = pca.transform(test_scaled)

    # Predict probabilities using trained models and save results
    id_codes = test['ID_code']
    results_df = pd.DataFrame({'ID_code': id_codes})

    # Assuming you want to make predictions using all models
    for name, model in models.items():
        predictions = model.predict_proba(test_pca)[:, 1]
        results_df[f'{name}_predicted_probabilities'] = predictions

    # Save the results to a CSV file
    results_df.to_csv('test_predictions.csv', index=False)

    # Display the first few rows of the results for verification
    print(results_df.head())

# Visualize PCA results
def visualize_pca_results(pca, X_train_scaled, y_train):
    # Explained variance plot
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.grid()
    plt.show()

    # 2D visualization of the first two principal components
    pca_2d = PCA(n_components=2)
    X_train_pca_2d = pca_2d.fit_transform(X_train_scaled)
    pca_df = pd.DataFrame(X_train_pca_2d, columns=['PC1', 'PC2'])
    pca_df['target'] = y_train.values

    plt.figure(figsize=(10, 6))
    plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['target'], cmap='viridis', alpha=0.5)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA - 2 Component Visualization')
    plt.colorbar(label='Target')
    plt.grid()
    plt.show()

    # Pairplot for the first four components
    pca_4d = PCA(n_components=4)
    X_train_pca_4d = pca_4d.fit_transform(X_train_scaled)
    pca_df_4d = pd.DataFrame(X_train_pca_4d, columns=['PC1', 'PC2', 'PC3', 'PC4'])
    pca_df_4d['target'] = y_train.values

    sns.pairplot(pca_df_4d, hue='target', diag_kind='kde')
    plt.show()


# Clean data
df = clean_data(data)

# Prepare features and target
X = df.drop(columns=['target']).values
y = df['target']

# Train and evaluate models using StratifiedKFold
models, scaler, pca = train_and_evaluate_models(X, y)

# Apply trained models to the test dataset
make_test_predictions(models, scaler, pca)

