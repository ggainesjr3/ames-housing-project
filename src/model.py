import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
import joblib

def preprocess_ames(df):
    """
    Standardizes schema and handles ordinal quality mappings.
    """
    df = df.copy()
    df.columns = [c.replace(' ', '') for c in df.columns]
    
    # 1. Drop high-cardinality noise
    noise_cols = ['Order', 'PID', 'Id']
    df = df.drop(columns=[c for c in noise_cols if c in df.columns])

    # 2. Ordinal Mapping: Converting quality labels to a numeric hierarchy
    qual_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
    qual_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu']
    for col in qual_cols:
        if col in df.columns:
            df[col] = df[col].fillna('None').map(qual_map)

    # 3. Log Transformation: Handling price skewness
    if 'SalePrice' in df.columns:
        df['SalePrice'] = np.log1p(df['SalePrice'])

    # 4. One-Hot Encoding for remaining categorical data
    df = pd.get_dummies(df).fillna(0)
    return df

if __name__ == "__main__":
    TRAIN_PATH = '/home/gary/ames-housing-project/data/train.csv'
    MODEL_SAVE_PATH = '/home/gary/ames-housing-project/src/ames_model.pkl'
    
    try:
        # DATA BOUNCER: Check if the ingredients are valid
        if not os.path.exists(TRAIN_PATH):
            raise FileNotFoundError(f"Missing file: {TRAIN_PATH}")
            
        raw_data = pd.read_csv(TRAIN_PATH)
        if raw_data.empty or raw_data.shape[1] < 2:
            raise ValueError("Bouncer Check Failed: train.csv is empty or corrupted.")
            
        print(f"📂 [SYSTEM_LOG] Data Loaded: {raw_data.shape[0]} rows")
        
        # Preprocessing & Splitting
        df = preprocess_ames(raw_data)
        X = df.drop('SalePrice', axis=1)
        y = df['SalePrice']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # XGBoost Configuration with Early Stopping
        model = XGBRegressor(
            n_estimators=1000, 
            learning_rate=0.05, 
            max_depth=5, 
            n_jobs=-1,
            early_stopping_rounds=50
        )
        
        print("🏗️  [BUILD_LOG] Engineering Ames Regression Engine...")
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        # Performance Evaluation
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        print(f"\n✅ [METRICS_LOG] Performance Verified - Log-RMSE: {rmse:.4f}")
        
        # 📊 [ANALYSIS] Feature Importance
        importance = pd.Series(model.feature_importances_, index=X.columns)
        print("\n📈 [ANALYSIS] Top 5 Price Drivers:")
        print(importance.sort_values(ascending=False).head(5))
        
        # 🍾 [SAVE_LOG] Persistent Storage
        joblib.dump(model, MODEL_SAVE_PATH)
        print(f"\n🧠 [SYSTEM_LOG] Model Brain saved to: {MODEL_SAVE_PATH}")
        
    except Exception as e:
        print(f"\n❌ [CRITICAL_ERROR]: {e}")
