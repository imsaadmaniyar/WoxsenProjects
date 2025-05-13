import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load the dataset to get feature structure
df = pd.read_csv("facebookadcampaigndataset.csv")
df = df.dropna(subset=['approved_conversion', 'spent'])
df = df[df['spent'] > 0].copy()
df['revenue'] = df['approved_conversion'] * 100
df['roas'] = df['revenue'] / df['spent']
df['ctr'] = df['clicks'] / df['impressions'].replace(0, 1)
df['cpm'] = (df['spent'] / df['impressions'].replace(0, 1)) * 1000
df['is_successful'] = (df['roas'] > 1).astype(int)
df_encoded = pd.get_dummies(df, columns=['age', 'gender'], drop_first=True)
X = df_encoded.drop(columns=[
    'ad_id', 'reporting_start', 'reporting_end',
    'campaign_id', 'fb_campaign_id', 'revenue', 'roas', 'is_successful'
])
y = df_encoded['is_successful']

# Train model (or load from file if saved)
model = RandomForestClassifier(random_state=42, n_jobs=-1)
model.fit(X, y)

st.title("Facebook Ad Campaign Success Predictor")

# Inputs from user
impressions = st.number_input("Impressions", min_value=0)
clicks = st.number_input("Clicks", min_value=0)
spent = st.number_input("Spent ($)", min_value=0.0, format="%f")
approved_conversion = st.number_input("Approved Conversions", min_value=0)
age = st.selectbox("Age Group", df['age'].unique())
gender = st.selectbox("Gender", df['gender'].unique())

if st.button("Predict Success"):
    # Derived features
    revenue = approved_conversion * 100
    ctr = clicks / impressions if impressions > 0 else 0
    cpm = (spent / impressions * 1000) if impressions > 0 else 0

    # Create input DataFrame
    input_data = {
        'impressions': impressions,
        'clicks': clicks,
        'spent': spent,
        'approved_conversion': approved_conversion,
        'ctr': ctr,
        'cpm': cpm,
    }

    # One-hot encode age and gender
    for col in X.columns:
        if col.startswith("age_"):
            input_data[col] = 1 if col == f"age_{age}" else 0
        elif col.startswith("gender_"):
            input_data[col] = 1 if col == f"gender_{gender}" else 0

    # Add missing features as 0 (if any)
    for col in X.columns:
        if col not in input_data:
            input_data[col] = 0

    input_df = pd.DataFrame([input_data])[X.columns]  # Ensure order
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][prediction]

    if prediction == 1:
        st.success(f"The campaign is predicted to be SUCCESSFUL with {proba*100:.2f}% confidence.")
    else:
        st.error(f"The campaign is predicted to be UNSUCCESSFUL with {proba*100:.2f}% confidence.")
