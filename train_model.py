import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# 1. Load Data from the stable official source
url = "http://jse.amstat.org/v19n3/decock/AmesHousing.txt"

try:
    # Use sep='\t' for the official tab-separated file
    df = pd.read_csv(url, sep='\t')
    print("✅ Data loaded successfully!")
except Exception as e:
    print(f"Error loading data: {e}")
    # Fallback to CSV if the official source is down
    backup_url = "https://raw.githubusercontent.com/dataprofessor/data/master/AmesHousing.csv"
    df = pd.read_csv(backup_url)

# 2. Clean column names (removes spaces like 'Gr Liv Area' -> 'GrLivArea')
df.columns = df.columns.str.replace(' ', '')

# 3. Feature Engineering: Construction & Age
df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF'].fillna(0)
df['HouseAge'] = df['YrSold'] - df['YearBuilt']

# 4. Feature Engineering: Quality Mapping
qual_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
df['ExterQualNum'] = df['ExterQual'].map(qual_map).fillna(0)

# 5. Neighborhood One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['Neighborhood'], prefix='NBH', dtype=int)

# 6. Select Final Features
nbh_cols = [col for col in df_encoded.columns if col.startswith('NBH_')]
base_features = ['TotalSF', 'OverallQual', 'ExterQualNum', 'HouseAge', 'GarageCars']

# Handle any remaining NaNs
df_encoded[base_features] = df_encoded[base_features].fillna(0)

final_features = base_features + nbh_cols
X = df_encoded[final_features]

# 7. Log Transform the Target (SalePrice)
# This makes the model predict the percentage impact rather than flat dollars
y = np.log1p(df_encoded['SalePrice'])

# 8. Train the Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# 9. Calculate Feature Importance
# We look at the absolute value of coefficients to see "strength"
importance = np.abs(model.coef_)
feature_importance = pd.Series(importance, index=final_features).sort_values(ascending=False)

# 10. Save all artifacts for the Streamlit App
joblib.dump(model, 'ames_log_model.joblib')
joblib.dump(final_features, 'model_features.joblib')
joblib.dump(df['Neighborhood'].unique().tolist(), 'neighborhoods.joblib')
joblib.dump(feature_importance.head(10), 'feature_importance.joblib')

print(f"🚀 Training Complete!")
print(f"Model saved with {len(final_features)} features.")
print("Feature importance data generated for the top 10 predictors.")