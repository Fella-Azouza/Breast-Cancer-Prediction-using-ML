import streamlit as st
import joblib
import pandas as pd
import numpy as np  # Import numpy to avoid the NameError

# Load the trained model (Random Forest)
model = joblib.load('rf_model.pkl')  # Load the saved model

# Load the custom CSS from the file
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# App Title
st.markdown('<div class="title">Breast Cancer</div>', unsafe_allow_html=True) 

# App Description
st.markdown("""
    This app predicts whether a patient has breast cancer based on various features.
    Please input the tumor characteristics below to classify it as either benign or malignant.
""", unsafe_allow_html=True)

# Input fields with a form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        symmetry_mean = st.text_input('Symmetry Mean')
        fractal_dimension_mean = st.text_input('Fractal Dimension Mean')
        radius_se = st.text_input('Radius SE')
        texture_se = st.text_input('Texture SE')
        smoothness_se = st.text_input('Smoothness SE')
        compactness_se = st.text_input('Compactness SE')

    with col2:
        concave_points_se = st.text_input('Concave Points SE')
        symmetry_se = st.text_input('Symmetry SE')
        texture_worst = st.text_input('Texture Worst')
        smoothness_worst = st.text_input('Smoothness Worst')
        concave_points_worst = st.text_input('Concave Points Worst')
        symmetry_worst = st.text_input('Symmetry Worst')

    # Add a submit button with styling and center it using the .center-button class
    submit_button = st.form_submit_button('Classify Tumor', help="Click to classify the tumor", use_container_width=True)

# If button is clicked
if submit_button:
    # Create a DataFrame from the user input
    user_input_df = pd.DataFrame({
        'symmetry_mean': [symmetry_mean],
        'fractal_dimension_mean': [fractal_dimension_mean],
        'radius_se': [radius_se],
        'texture_se': [texture_se],
        'smoothness_se': [smoothness_se],
        'compactness_se': [compactness_se],
        'concave points_se': [concave_points_se],
        'symmetry_se': [symmetry_se],
        'texture_worst': [texture_worst],
        'smoothness_worst': [smoothness_worst],
        'concave points_worst': [concave_points_worst],
        'symmetry_worst': [symmetry_worst]
    })

    # Ensure the column names are exactly as they were when the model was trained
    expected_columns = [
        'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se',
        'smoothness_se', 'compactness_se', 'concave points_se', 'symmetry_se',
        'texture_worst', 'smoothness_worst', 'concave points_worst', 'symmetry_worst'
    ]

    try:
        user_input_df = user_input_df[expected_columns].apply(pd.to_numeric, errors='raise')

        # Predict the result using the loaded model
        prediction = model.predict(user_input_df)

        # Check if the prediction is a numpy array and handle it properly
        prediction_value = prediction[0] if isinstance(prediction, np.ndarray) else prediction

        # Display the prediction result
        if prediction_value == 0:
            st.success("The breast cancer is classified as **Benign** (No Cancer).")
        else:
            st.error("The breast cancer is classified as **Malignant**.")
    except ValueError:
        st.error("Please enter valid numerical values for all fields.")
