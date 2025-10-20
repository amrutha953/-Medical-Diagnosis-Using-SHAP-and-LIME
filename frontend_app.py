import streamlit as st
from PIL import Image
import requests
import io

st.set_page_config(page_title="XAI Medical Diagnosis", layout="wide")
st.title("🩺 Explainable AI for Medical Diagnosis (SHAP & LIME)")

st.write("Upload a medical image (e.g., Chest X-Ray) to visualize predictions with SHAP and LIME explainability methods.")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    if st.button("Run Diagnosis"):
        with st.spinner("Analyzing with model..."):
            files = {"image": uploaded_file.getvalue()}
            response = requests.post("http://127.0.0.1:5000/api/predict", files=files)
            data = response.json()

        st.subheader("🔍 Prediction Results")
        st.write(f"**Diagnosis:** {data['label']}")
        st.write(f"**Confidence:** {data['confidence']*100:.2f}%")

        col1, col2 = st.columns(2)
        with col1:
            st.image(data["shap_heatmap"], caption="SHAP Heatmap")
        with col2:
            st.image(data["lime_heatmap"], caption="LIME Heatmap")

        st.success("Explainability visualizations generated successfully!")

st.markdown("---")
st.caption("Developed using Streamlit + PyTorch/TensorFlow + SHAP + LIME")
