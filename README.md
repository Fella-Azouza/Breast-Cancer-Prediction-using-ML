# Breast Cancer Classification App
## Project Overview
A web application built with Streamlit that predicts whether a tumor is **benign** or **malignant** based on user input features. It allows users to input various tumor features, and it uses a pre-trained machine learning model to classify the tumor.

## Features
- Input form for user to enter various tumor features.
- Integration with a machine learning model for tumor classification.
- Display of tumor classification on the web page.

## Tech Stack
- Scikit-learn: A machine learning library for building and training the model.
- Streamlit: For creating an intuitive user interface and hosting the app.
- Joblib: For saving and loading the pre-trained machine learning model.
- Pandas and Numpy: For data preprocessing and manipulation.

## Model Selection

During the development process, we experimented with several machine learning models, including Supervised Learning models (Logistic Regression, Random Forest, KNN) and Unsupervised Learning models (K-Means Clustering, Hierarchical Clustering (Agglomerative Clustering), GMM). After evaluating their performance, we selected **Random Forest** as the final model due to its superior accuracy and robustness in predicting breast cancer outcomes.

## How to Run the App Locally

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/Fella-Azouza/Breast-Cancer-Prediction-using-ML.git
   cd Breast-Cancer-Prediction-using-ML
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. Open your browser and navigate to `http://localhost:8501` to use the app.





