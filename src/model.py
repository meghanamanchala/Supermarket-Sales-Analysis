import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set style
sns.set_theme(style="whitegrid")

def train_and_evaluate_model(input_path, output_dir):
    print(f"Loading cleaned dataset for model training: {input_path}")
    df = pd.read_csv(input_path)
    
    # Define features and target
    # We want to predict sales. We include unit_price and quantity as they are primary mathematical drivers,
    # but we also include categorical variables (branch, customer_type, gender, product_line, payment)
    # and temporal features (hour, day_of_week) to see if the model can capture the full structure.
    features = ['unit_price', 'quantity', 'branch', 'customer_type', 'gender', 'product_line', 'payment', 'hour', 'day_of_week']
    target = 'sales'
    
    X = df[features]
    y = df[target]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # Define categorical and numerical features for column transformer
    categorical_cols = ['branch', 'customer_type', 'gender', 'product_line', 'payment']
    numerical_cols = ['unit_price', 'quantity', 'hour', 'day_of_week']
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
        ]
    )
    
    # Create Random Forest pipeline
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10))
    ])
    
    # Create Linear Regression pipeline as baseline
    lr_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    # Fit models
    print("Training models...")
    rf_pipeline.fit(X_train, y_train)
    lr_pipeline.fit(X_train, y_train)
    
    # Save the Random Forest pipeline to models directory
    models_dir = os.path.join(os.path.dirname(input_path), "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "sales_prediction_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(rf_pipeline, f)
    print(f"Saved trained Random Forest model pipeline to: {model_path}")
    
    # Predictions
    y_pred_rf = rf_pipeline.predict(X_test)
    y_pred_lr = lr_pipeline.predict(X_test)
    
    # Evaluate Random Forest
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    
    # Evaluate Linear Regression
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)
    
    print("\n=== Model Performance: Linear Regression (Baseline) ===")
    print(f"RMSE: {rmse_lr:.4f}")
    print(f"MAE : {mae_lr:.4f}")
    print(f"R²  : {r2_lr:.4f}")
    
    print("\n=== Model Performance: Random Forest Regressor ===")
    print(f"RMSE: {rmse_rf:.4f}")
    print(f"MAE : {mae_rf:.4f}")
    print(f"R²  : {r2_rf:.4f}")
    
    # Save performance metrics to a dictionary
    metrics = {
        'Linear Regression': {'rmse': rmse_lr, 'mae': mae_lr, 'r2': r2_lr},
        'Random Forest': {'rmse': rmse_rf, 'mae': mae_rf, 'r2': r2_rf}
    }
    
    # Visualizations
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Actual vs Predicted Plot (Random Forest)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred_rf, alpha=0.6, color='#2ca02c')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title('Actual vs. Predicted Sales (Random Forest Regressor)', pad=15)
    plt.xlabel('Actual Sales (USD)')
    plt.ylabel('Predicted Sales (USD)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_actual_vs_predicted.png'), dpi=150)
    plt.close()
    
    # 2. Feature Importance Plot (Random Forest)
    # Get feature names after one-hot encoding
    cat_encoder = rf_pipeline.named_steps['preprocessor'].named_transformers_['cat']
    encoded_cat_cols = cat_encoder.get_feature_names_out(categorical_cols).tolist()
    feature_names = numerical_cols + encoded_cat_cols
    
    importances = rf_pipeline.named_steps['regressor'].feature_importances_
    feat_importances = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feat_importances = feat_importances.sort_values(by='importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=feat_importances.head(10), x='importance', y='feature', palette='viridis')
    plt.title('Top 10 Feature Importances (Random Forest Regressor)', pad=15)
    plt.xlabel('Relative Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_feature_importances.png'), dpi=150)
    plt.close()
    
    print("\n--- Top Feature Importances ---")
    print(feat_importances.head(10))
    
    return metrics, feat_importances

if __name__ == "__main__":
    input_file = os.path.join("dataset", "clean_supermarket_sales.csv")
    output_directory = "images"
    train_and_evaluate_model(input_file, output_directory)
