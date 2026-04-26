import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the model and the feature list
model = joblib.load('ames_log_model.joblib')
model_features = joblib.load('model_features.joblib') # This is the key!

st.title("🏡 Ames House Price Predictor")

# 2. Input fields (ensure these cover the main features)
sq_ft = st.number_input("Total Square Feet", value=1500)
year_built = st.slider("Year Built", 1850, 2026, 2000)
# ... add other inputs as needed ...

if st.button("Predict Price"):
    # 3. Create a dictionary with ALL features, initialized to 0
    # This ensures columns like 'Neighborhood_StoneBr' exist if they were in training
    input_dict = {feature: 0 for feature in model_features}
    
    # 4. Fill in the values from your UI
    # IMPORTANT: Use the EXACT names used in your training dataframe
    input_dict['GrLivArea'] = sq_ft
    input_dict['YearBuilt'] = year_built
    # input_dict['BedroomAbvGr'] = bedrooms ... etc
    
    # 5. Convert to DataFrame with the correct column order
    input_df = pd.DataFrame([input_dict])[model_features]
    
    # 6. Predict and reverse the Log
    log_prediction = model.predict(input_df)[0]
    final_price = np.exp(log_prediction)
    
    st.success(f"### Estimated Price: ${final_price:,.2f}")