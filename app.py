import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="High Profitability Classifier", layout="wide")

st.title("Nifty 500 High Profitability Classifier")
st.markdown("### ML Assignment 2")
st.write("This app predicts whether a company is 'Highly Profitable' (Top 25% Net Profit) based on its quarterly operating metrics.")

# Model Paths
model_paths = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest (Ensemble)": "model/random_forest.pkl"
}

# Sidebar for inputs
st.sidebar.header("1. Upload Test Data")
uploaded_file = st.sidebar.file_uploader("Upload test_data.csv", type=["csv"])

st.sidebar.header("2. Select Model")
selected_model = st.sidebar.selectbox("Choose a Classification Model", list(model_paths.keys()))

if uploaded_file is not None:
    # Load Data
    data = pd.read_csv(uploaded_file)
    st.write("### 📊 Uploaded Test Data Preview")
    st.dataframe(data.head())
    
    if 'Target' not in data.columns:
        st.error("Error: The uploaded CSV must contain the 'Target' column for evaluation.")
    else:
        X_test = data.drop(columns=['Target'])
        y_test = data['Target']
        
        # Load Model and Scaler
        try:
            model = joblib.load(model_paths[selected_model])
            scaler = joblib.load("model/scaler.pkl")
            
            # Align features and scale
            X_test_scaled = scaler.transform(X_test.fillna(0))
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            try:
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
                auc = roc_auc_score(y_test, y_prob)
            except:
                auc = roc_auc_score(y_test, y_pred)
                
            # Display Metrics
            st.write(f"### 📈 Evaluation Metrics for {selected_model}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
            c2.metric("AUC Score", f"{auc:.4f}")
            c3.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.4f}")
            
            c4, c5, c6 = st.columns(3)
            c4.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.4f}")
            c5.metric("F1 Score", f"{f1_score(y_test, y_pred, zero_division=0):.4f}")
            c6.metric("MCC Score", f"{matthews_corrcoef(y_test, y_pred):.4f}")
            
            # Confusion Matrix
            st.write("### 🧩 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(4,3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                        xticklabels=['Not Highly Profitable', 'Highly Profitable'],
                        yticklabels=['Not Highly Profitable', 'Highly Profitable'])
            plt.ylabel('Actual Label')
            plt.xlabel('Predicted Label')
            st.pyplot(fig)
            
        except FileNotFoundError:
            st.error("Model files not found. Please run train_models.py locally first to generate the models.")
else:
    st.info("👈 Please upload the 'test_data.csv' file from the sidebar to view model evaluations.")