
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.title("📊 Facebook Ad Campaign ROAS Predictor")

uploaded_file = st.file_uploader("📂 Upload your Facebook Ad Campaign CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Raw Data Preview")
    st.dataframe(df.head())

    # Data Cleaning
    df = df.dropna(subset=["approved_conversion", "spent"])
    df = df[df["spent"] > 0]
    df["revenue"] = df["approved_conversion"] * 100
    df["roas"] = df["revenue"] / df["spent"]
    df["ctr"] = df["clicks"] / df["impressions"].replace(0, 1)
    df["cpm"] = (df["spent"] / df["impressions"].replace(0, 1)) * 1000

    # One-hot encoding
    df_encoded = pd.get_dummies(df, columns=["age", "gender"], drop_first=True)

    # Drop unneeded columns
    drop_cols = [
        "ad_id", "reporting_start", "reporting_end",
        "campaign_id", "fb_campaign_id", "revenue", "approved_conversion"
    ]
    df_encoded.drop(columns=[col for col in drop_cols if col in df_encoded.columns], inplace=True)

    # Model prep
    X = df_encoded.drop(columns=["roas"])
    y = df_encoded["roas"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.success("✅ Model trained successfully!")
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")

    # --- Manual Input for Prediction ---
    st.header("🎯 Predict ROAS for a New Ad Campaign")

    with st.form("prediction_form"):
        impressions = st.number_input("Impressions", min_value=1, value=1000)
        clicks = st.number_input("Clicks", min_value=0, value=100)
        spent = st.number_input("Amount Spent ($)", min_value=1.0, value=50.0)
        age_group = st.selectbox("Age Group", df["age"].unique())
        gender = st.selectbox("Gender", df["gender"].unique())

        submit = st.form_submit_button("Predict ROAS")

    if submit:
        ctr = clicks / impressions
        cpm = (spent / impressions) * 1000

        input_data = {
            "impressions": impressions,
            "clicks": clicks,
            "spent": spent,
            "ctr": ctr,
            "cpm": cpm,
        }

        # Add encoded age and gender fields
        for col in X.columns:
            if "age_" in col:
                input_data[col] = 1 if f"age_{age_group}" == col else 0
            elif "gender_" in col:
                input_data[col] = 1 if f"gender_{gender}" == col else 0
            elif col not in input_data:
                input_data[col] = 0  # Default value

        input_df = pd.DataFrame([input_data])
        input_df = input_df.reindex(columns=X.columns, fill_value=0)

        prediction = model.predict(input_df)[0]
        st.success(f"💡 Predicted ROAS: **{prediction:.2f}**")
