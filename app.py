import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import numpy as np
import plotly.graph_objects as go

# Load the Gradient Boosting model pipeline for text data predictions
try:
    gb_model_pipeline = joblib.load('optimized_model_pipeline.pkl')
except FileNotFoundError:
    st.error("Model file 'optimized_model_pipeline.pkl' not found. Please check the file path.")

# Define calculate_disease_spread function for image input
def calculate_disease_spread(image_array, lower_threshold=50, upper_threshold=200):
    affected_area = np.sum((image_array > lower_threshold) & (image_array < upper_threshold))
    disease_spread_score = affected_area / (image_array.shape[0] * image_array.shape[1])

    # Categorize spread score
    if disease_spread_score < 0.1:
        spread_category = 'Low'
    elif disease_spread_score < 0.3:
        spread_category = 'Medium'
    elif disease_spread_score < 0.5:
        spread_category = 'High'
    else:
        spread_category = 'Severe'
    
    return disease_spread_score, spread_category

# Define Streamlit app interface with dark blue theme
st.set_page_config(page_title="Fibro-Vision - IPF Progression Detection", layout="wide")

# Set custom background colors and images
st.markdown("""
    <style>
        body {
            background-color: #001f3d;
            font-family: 'Arial', sans-serif;
        }
        .title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #ffffff;
            padding: 20px;
            background-color: #004080;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .header {
            text-align: center;
            font-size: 28px;
            font-weight: bold;
            color: #ffffff;
            padding: 15px;
            background-color: #003366;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .section {
            padding: 20px;
        }
        .input-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            padding: 20px;
        }
        .input-container input, .input-container select {
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #ddd;
            font-size: 16px;
        }
        .input-container input[type="number"] {
            width: 100%;
        }
        .sidebar .sidebar-content {
            background-color: #003366;
            padding-top: 50px;
        }
        .sidebar .sidebar-content a {
            color: #ffffff;
            text-decoration: none;
            padding: 12px;
            font-size: 18px;
            display: block;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .sidebar .sidebar-content a:hover {
            background-color: #004080;
            color: #ffffff;
        }
        .sidebar .sidebar-content .selected {
            background-color: #004080;
            color: #ffffff;
        }
        .button {
            background-color: #004080;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            font-weight: bold;
            transition: background-color 0.3s;
        }
        .button:hover {
            background-color: #003366;
        }
    </style>
""", unsafe_allow_html=True)

# Create a custom top navigation bar
menu = ["Home", "Clinical Data Prediction", "CT Scan Prediction"]
st.sidebar.markdown("<h1 style='color:white;'>Navigation</h1>", unsafe_allow_html=True)
selection = st.sidebar.radio("Select a Page", menu)

# Home page content
if selection == "Home":
    st.markdown("<div class='title'>FIBRO-VISION: OPENCV POWERED INSIGHTS FOR IPF PROGRESSION</div>", unsafe_allow_html=True)
    st.image("bg1.jpg", use_column_width=True)  # Replace with the actual image path
    st.write("""
    Welcome to **Fibro-Vision**, a cutting-edge tool powered by OpenCV and machine learning to analyze IPF progression.
    Select a category from the sidebar to get started:
    - Clinical Data Prediction
    - CT Scan Image Prediction
    """)

# Clinical Data Prediction Page
elif selection == "Clinical Data Prediction":
    st.markdown("<div class='header'>Predict IPF Spread from Clinical Data</div>", unsafe_allow_html=True)
    st.markdown("Use the form below to input clinical data for IPF progression prediction.")

    # Create input fields for numerical data
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            weeks = st.number_input("Weeks:", min_value=0, max_value=100, value=10)
        with col2:
            percent = st.number_input("Percent:", min_value=0.0, max_value=100.0, value=50.0)

        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age:", min_value=0, max_value=100, value=30)
        with col2:
            sex = st.selectbox("Sex:", options=["Male", "Female"])

        smoking_status = st.selectbox("Smoking Status:", options=["Never", "Ex-smoker", "Smoker"])

    # Prepare the input data as a DataFrame
    input_data = pd.DataFrame({
        'Weeks': [weeks],
        'Percent': [percent],
        'Age': [age],
        'Sex': [sex],
        'SmokingStatus': [smoking_status]
    })

    # Predict with the Gradient Boosting model
    if st.button("Predict IPF Spread from Clinical Data", use_container_width=True):
        if 'gb_model_pipeline' in locals():
            try:
                prediction = gb_model_pipeline.predict(input_data)
                st.write(f"Predicted FVC (Forced Vital Capacity): {prediction[0]:.2f} (IPF Spread Estimate)", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction error: {e}")
        else:
            st.error("Model pipeline not loaded. Check for 'optimized_model_pipeline.pkl'.")

# CT Scan Prediction Page
elif selection == "CT Scan Prediction":
    st.markdown("<div class='header'>Predict IPF Spread from CT Scan Image</div>", unsafe_allow_html=True)
    st.markdown("Upload a CT Scan image to predict disease spread severity.")

    # Upload an image file
    uploaded_image = st.file_uploader("Upload CT Scan Image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        # Load and preprocess the image
        image = Image.open(uploaded_image).convert('L')  # Convert to grayscale
        image = image.resize((128, 128))  # Resize to match model's input size if needed
        image_array = np.array(image)  # Convert image to numpy array

        # Predict disease spread with the calculate_disease_spread function
        if st.button("Predict IPF Spread from Image", use_container_width=True):
            spread_score, spread_category = calculate_disease_spread(image_array)  # Calculate spread score and severity
            st.write(f"Disease Spread Score: {spread_score:.4f}")
            st.write(f"Disease Severity Level: {spread_category}")
            st.image(image, caption="Uploaded CT Scan Image", use_column_width=True)
            
            # Plotting the affected area using Plotly
            fig = go.Figure(data=go.Heatmap(z=image_array, colorscale='Hot', colorbar=dict(title="Intensity")))
            fig.update_layout(title="CT Scan Image Intensity Heatmap")
            st.plotly_chart(fig, use_container_width=True)
