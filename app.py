import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import cv2

import streamlit as st

# Set Streamlit page configuration
st.set_page_config(page_title="FIBRO-VISION - IPF PROGRESSION DETECTION", layout="wide")

# Custom CSS for styling the page
st.markdown("""
    <style>
        body {
            background-color: #001f3d;
            font-family: 'Arial', sans-serif;
        }
        .header {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #ffffff;
            padding: 20px;
            background-color: #004080;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .subheader {
            text-align: center;
            font-size: 28px;
            font-weight: bold;
            color: #ffffff;
            padding: 15px;
            background-color: #003366;
            border-radius: 10px;
            margin-bottom: 20px;
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

# Example of using custom CSS styles
st.markdown("<h1 class='header'>Fibro-Vision OpenCV Predict IPF Progression</h1>", unsafe_allow_html=True)

# Further content...


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
        color = 'green'
    elif disease_spread_score < 0.3:
        spread_category = 'Medium'
        color = 'yellow'
    elif disease_spread_score < 0.5:
        spread_category = 'High'
        color = 'orange'
    else:
        spread_category = 'Severe'
        color = 'red'
    
    return disease_spread_score, spread_category, color



# Create navigation bar with bold styling
menu = ["HOME", "CLINICAL DATA PREDICTION", "CT SCAN PREDICTION"]
st.sidebar.markdown("<h1 class='navigator'>NAVIGATION</h1>", unsafe_allow_html=True)

# Adding custom CSS to make the text bold
st.sidebar.markdown("""
    <style>
        .navigator {
            font-weight: bold; /* Make the title bold */
            font-size: 24px; /* Increase the font size for better visibility */
        }
        .stRadio>div>label {
            font-weight: bold; /* Make the radio button options bold */
            font-size: 25px; /* Adjust font size for radio button options */
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar radio buttons
selection = st.sidebar.radio("", menu, key="nav")


# Home page content
if selection == "HOME":
    st.markdown("<h1 style='text-align: center; color: white;'>FIBRO-VISION</h1>", unsafe_allow_html=True)
    with st.container():
        col1, col2 = st.columns([2, 3])
        with col1:
            st.image("bg1.jpg", caption="", use_column_width=True)
        with col2:
            st.write("""
    ## Welcome to **Fibro-Vision**
    Fibro-Vision is a cutting-edge tool powered by OpenCV and machine learning to analyze the progression of **Idiopathic Pulmonary Fibrosis (IPF)**, 
    offering valuable insights for clinicians and researchers.

    ### Features:
    - **Clinical Data Prediction:** Provide patient clinical data (e.g., age, smoking history, and lung function tests) to predict the severity and progression of IPF.
    - **CT Scan Image Prediction:** Upload CT scan images to assess the spread and severity of IPF through advanced image analysis.

    ### How to Get Started:
    Select a category from the **sidebar** to explore our functionalities:
    - **Clinical Data Prediction**: Analyze clinical data for disease progression.
    - **CT Scan Image Prediction**: Upload images to receive insights based on image processing.
""")

# Clinical Data Prediction Page
elif selection == "CLINICAL DATA PREDICTION":
    st.markdown("<div class='header'>Predict IPF Spread from Clinical Data</div>", unsafe_allow_html=True)
    st.markdown("Use the form below to input clinical data for IPF progression prediction.")

    # Create input fields for numerical data
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            weeks = st.number_input("Weeks:", min_value=0, max_value=100, value=0)
        with col2:
            percent = st.number_input("Percent:", min_value=0.0, max_value=100.0, value=0.00)

        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age:", min_value=0, max_value=100, value=0)
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
elif selection == "CT SCAN PREDICTION":
    st.markdown("<h2 style='text-align: center; color: white;'>Predict IPF Spread from CT Scan Image</h2>", unsafe_allow_html=True)
    
    # File uploader for CT Scan image
    uploaded_image = st.file_uploader("Upload CT Scan Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_image is not None:
        # Load and preprocess the image
        image = Image.open(uploaded_image).convert('L')  # Convert to grayscale
        image = image.resize((120, 120))  # Resize slightly smaller for better visual balance
        image_array = np.array(image)  # Convert image to numpy array

        # Predict disease spread
        if st.button("Predict IPF Spread from Image"):
            spread_score, spread_category, color = calculate_disease_spread(image_array)

            # Display prediction results
            st.markdown(f"<p style='font-size:20px; color: green; text-align:center;'><b>Disease Spread Score:</b> {spread_score:.4f}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:20px; color:{color}; text-align:center;'><b>Disease Severity Level:</b> {spread_category}</p>", unsafe_allow_html=True)

            # Custom color maps for severity
            severity_colormap = {
                'Low': [(0, 'green')],
                'Medium': [(0, 'yellow')],
                'High': [(0, 'orange')],
                'Severe': [(0, 'red')],
            }

            # Display the grayscale image
            st.markdown("<h4 style='text-align: center; color: white;'>Grayscale CT Scan Image</h4>", unsafe_allow_html=True)
            st.image(image, caption="Grayscale CT Scan Image", width=500, use_column_width=False)

            # Highlighting disease spread (Overlay heatmap on image)
            heatmap = np.zeros_like(image_array, dtype=np.float32)

            # Simulate spread areas based on spread score (for demonstration purposes)
            if spread_category == "Low":
                heatmap += 0.1
            elif spread_category == "Medium":
                heatmap += 0.3
            elif spread_category == "High":
                heatmap += 0.6
            elif spread_category == "Severe":
                heatmap += 1.0

            # Custom color for severity (mapped to a custom color scheme)
            if spread_category == "Low":
                custom_color = np.array([0, 255, 0])  # Green
            elif spread_category == "Medium":
                custom_color = np.array([255, 255, 0])  # Yellow
            elif spread_category == "High":
                custom_color = np.array([255, 165, 0])  # Orange
            elif spread_category == "Severe":
                custom_color = np.array([255, 0, 0])  # Red

            # Apply the custom color to the heatmap
            heatmap_colored = np.zeros((image_array.shape[0], image_array.shape[1], 3), dtype=np.uint8)
            for i in range(image_array.shape[0]):
                for j in range(image_array.shape[1]):
                    if heatmap[i, j] > 0.0:
                        heatmap_colored[i, j] = custom_color

            # Convert grayscale image to 3-channel RGB for proper blending
            image_rgb = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)

            # Combine the grayscale image and the heatmap
            highlighted_image = cv2.addWeighted(image_rgb, 0.7, heatmap_colored, 0.3, 0)

            # Convert the result to a format Streamlit can display
            highlighted_image_pil = Image.fromarray(highlighted_image)

            # Display the highlighted image
            st.markdown("<h4 style='text-align: center; color: white;'>Highlighted Disease Spread</h4>", unsafe_allow_html=True)
            st.image(highlighted_image_pil, caption="Highlighted Disease Spread", width=500, use_column_width=False)

            # Create and display heatmap for severity visualization
            fig = go.Figure(data=go.Heatmap(
                z=image_array,
                colorscale=severity_colormap[spread_category],
                colorbar=dict(title="Intensity")
            ))
            fig.update_layout(
                title=f"CT Scan Visualization ({spread_category} Severity)",
                title_x=0.5,
                width=500,
                height=500,
            )

            st.markdown("<h4 style='text-align: center; color: white;'>Severity Visualization Heatmap</h4>", unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)

            # Optional horizontal line for separation
            st.markdown("<hr style='border: 1px solid white;'>", unsafe_allow_html=True)
