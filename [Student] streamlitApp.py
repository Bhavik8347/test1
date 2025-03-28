# Import required libraries
import streamlit as st
import pickle
import numpy as np
import pandas as pd
from PIL import Image

# Set the page configuration
st.set_page_config(page_title="Timelytics", layout="wide")

# Title and description
st.title("Timelytics: Optimize your supply chain with advanced forecasting techniques.")
st.caption(
    "Timelytics is an ensemble model that utilizes XGBoost, Random Forest, and SVM "
    "to accurately forecast Order to Delivery (OTD) times."
)

# Load the trained ensemble model
modelfile = "./voting_model.pkl"

@st.cache_resource
def load_model():
    with open(modelfile, "rb") as f:
        model = pickle.load(f)
    return model

# Load the model
voting_model = load_model()

# Define wait time predictor function
def waitime_predictor(
    purchase_dow, purchase_month, year, product_size_cm3, product_weight_g, 
    geolocation_state_customer, geolocation_state_seller, distance
):
    prediction = voting_model.predict(np.array([[  
        purchase_dow, purchase_month, year, product_size_cm3, 
        product_weight_g, geolocation_state_customer, 
        geolocation_state_seller, distance
    ]]))
    return round(prediction[0])

# Sidebar inputs
with st.sidebar:
    img = Image.open("./assets/supply_chain_optimisation.jpg")
    st.image(img)
    
    st.header("Input Parameters")
    purchase_dow = st.number_input("Purchased Day of the Week", min_value=0, max_value=6, step=1, value=3)
    purchase_month = st.number_input("Purchased Month", min_value=1, max_value=12, step=1, value=1)
    year = st.number_input("Purchased Year", value=2018)
    product_size_cm3 = st.number_input("Product Size in cm^3", value=9328)
    product_weight_g = st.number_input("Product Weight in grams", value=1800)
    geolocation_state_customer = st.number_input("Geolocation State of the Customer", value=10)
    geolocation_state_seller = st.number_input("Geolocation State of the Seller", value=20)
    distance = st.number_input("Distance", value=475.35)

# Submit button
submit = st.button("Submit")

# Output section
with st.container():
    st.header("Output: Wait Time in Days")

    if submit:
        prediction = waitime_predictor(
            purchase_dow, purchase_month, year, product_size_cm3, 
            product_weight_g, geolocation_state_customer, 
            geolocation_state_seller, distance
        )
        with st.spinner(text="This may take a moment..."):
            st.write(f"**Predicted Wait Time:** {prediction} days")

# Sample dataset
data = {
    "Purchased Day of the Week": [0, 3, 1],
    "Purchased Month": [6, 3, 1],
    "Purchased Year": [2018, 2017, 2018],
    "Product Size in cm^3": [37206.0, 63714, 54816],
    "Product Weight in grams": [16250.0, 7249, 9600],
    "Geolocation State Customer": [25, 25, 25],
    "Geolocation State Seller": [20, 7, 20],
    "Distance": [247.94, 250.35, 4.915],
}

df = pd.DataFrame(data)

# Display dataset
st.header("Sample Dataset")
st.write(df)
