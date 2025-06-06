import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("Cleaned_Energy_Consumption.csv")

df = load_data()

st.title("⚡ Energy Consumption Predictor")
st.write("Upload cleaned energy data and predict consumption based on inputs.")

# Select features and target
X = df.drop(columns=["EnergyConsumption"])
y = df["EnergyConsumption"]

# Identify categorical columns
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)],
    remainder="passthrough"
)

# Create pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Input fields in Streamlit UI
st.sidebar.header("Enter Input Values")

def user_input():
    inputs = {}
    for col in X.columns:
        if col in categorical_cols:
            options = list(df[col].unique())
            inputs[col] = st.sidebar.selectbox(f"{col}", options)
        else:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            mean_val = float(df[col].mean())
            inputs[col] = st.sidebar.slider(f"{col}", min_val, max_val, mean_val)
    return pd.DataFrame([inputs])

user_df = user_input()

# Make prediction
if st.button("Predict Energy Consumption"):
    prediction = model.predict(user_df)
    st.success(f"🔋 Predicted Energy Consumption: {prediction[0]:.2f} units")

# Show dataset preview
with st.expander("📊 Show Raw Data"):
    st.dataframe(df)
