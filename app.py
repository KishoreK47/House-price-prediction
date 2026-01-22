import streamlit as st
import pandas as pd
import joblib

# Load model and features
model = joblib.load(r"C:\Users\kisho\Downloads\Clinch_mini_project\House_project\Model\house_price_lr_model.pkl")
features = joblib.load(r"C:\Users\kisho\Downloads\Clinch_mini_project\House_project\Model\model_features.pkl")

st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("🏠 House Price Prediction App")
st.write("Baseline Linear Regression Model")

# User input
area = st.number_input("Area (sq ft)", min_value=500, max_value=10000, step=100)
bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5])
bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4])
stories = st.selectbox("Stories", [1, 2, 3, 4])
parking = st.selectbox("Parking spaces", [0, 1, 2, 3])

mainroad = st.selectbox("Main Road", ["yes", "no"])
guestroom = st.selectbox("Guest Room", ["yes", "no"])
basement = st.selectbox("Basement", ["yes", "no"])
hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"])
airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])
prefarea = st.selectbox("Preferred Area", ["yes", "no"])
furnishingstatus = st.selectbox(
    "Furnishing Status",
    ["furnished", "semi-furnished", "unfurnished"]
)

# Convert inputs
input_dict = {
    'area': area,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'stories': stories,
    'parking': parking,
    'mainroad': 1 if mainroad == "yes" else 0,
    'guestroom': 1 if guestroom == "yes" else 0,
    'basement': 1 if basement == "yes" else 0,
    'hotwaterheating': 1 if hotwaterheating == "yes" else 0,
    'airconditioning': 1 if airconditioning == "yes" else 0,
    'prefarea': 1 if prefarea == "yes" else 0,
    'furnishingstatus_semi-furnished': 1 if furnishingstatus == "semi-furnished" else 0,
    'furnishingstatus_unfurnished': 1 if furnishingstatus == "unfurnished" else 0
}

# Create DataFrame
input_df = pd.DataFrame([input_dict])

# Ensure correct column order
input_df = input_df.reindex(columns=features, fill_value=0)

# Prediction
if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated House Price: ₹ {prediction:,.2f}")
