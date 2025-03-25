import streamlit as st
import pandas as pd
import joblib
import numpy as np
import torch
import plotly.express as px
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
st.set_page_config(page_title="Fake Job Detection System", page_icon="🕵️‍♂️")

st.title("Fake Job Detection System")
# Load models and preprocessing artifacts with caching
@st.cache_resource
def load_models():
    return {
        "stacking_model": joblib.load("model/fake_job_detection_model.pkl"),
        "scaler": joblib.load("model/scaler.pkl"),
        "label_encoders": joblib.load("model/label_encoders.pkl"),  # Dictionary of LabelEncoders
        "tokenizer": BertTokenizer.from_pretrained("bert-base-uncased"),
        "bert_model": BertModel.from_pretrained("bert-base-uncased").eval(),
    }

models = load_models()

# Function to compute BERT embeddings
@torch.no_grad()
def get_bert_embedding(text):
    tokens = models["tokenizer"](text, return_tensors="pt", truncation=True, padding="max_length", max_length=50)
    embedding = models["bert_model"](**tokens).last_hidden_state[:, 0, :].cpu().numpy().flatten()
    return (embedding - embedding.mean()) / (embedding.std() + 1e-6)  # Normalize embeddings

# Load dataset (cached for faster loading)
@st.cache_data
def load_data():
    df = pd.read_csv("fake_job_postings.csv").sample(n=15000, random_state=42)
    df.drop(columns=["job_id", "company_profile", "location"], errors="ignore", inplace=True)
    df.fillna("", inplace=True)

    # Balance real and fake job posts
    min_count = df["fraudulent"].value_counts().min()
    df_balanced = df.groupby("fraudulent").apply(lambda x: x.sample(n=min_count, random_state=42)).reset_index(drop=True)

    df_balanced["employment_type"] = df_balanced["employment_type"].astype("category")
    df_balanced["industry"] = df_balanced["industry"].astype("category")

    return df_balanced

df = load_data()

# UI Title
st.title("🔍 Fake Job Detection System")

# Sidebar Filters
st.sidebar.header("🔎 Filters")
employment_types = df["employment_type"].dropna().unique().tolist()
industries = df["industry"].dropna().unique().tolist()

selected_employment_type = st.sidebar.multiselect("Select Employment Type", employment_types, default=employment_types[:2])
selected_industry = st.sidebar.multiselect("Select Industry", industries, default=industries[:2])
fraud_filter = st.sidebar.radio("Show Only", ["All", "Fake Jobs", "Real Jobs"])

# Apply Filters
filtered_df = df.copy()
if selected_employment_type:
    filtered_df = filtered_df[filtered_df["employment_type"].isin(selected_employment_type)]
if selected_industry:
    filtered_df = filtered_df[filtered_df["industry"].isin(selected_industry)]
if fraud_filter == "Fake Jobs":
    filtered_df = filtered_df[filtered_df["fraudulent"] == 1]
elif fraud_filter == "Real Jobs":
    filtered_df = filtered_df[filtered_df["fraudulent"] == 0]

# Display Filtered Job Listings
st.write("### Filtered Job Listings")
st.dataframe(filtered_df[["title", "employment_type", "industry", "fraudulent"]], use_container_width=True)

# ✅ Bar Chart - Jobs per Industry
industry_counts = filtered_df["industry"].value_counts().reset_index()
industry_counts.columns = ["industry", "count"]

fig_bar = px.bar(
    industry_counts,
    x="industry",
    y="count",
    title="📊 Number of Jobs per Industry",
    labels={"industry": "Industry", "count": "Job Count"},
    text_auto=True
)
st.plotly_chart(fig_bar, use_container_width=True)

# Function to encode categorical variables safely
def encode_category(value, category):
    encoder = models["label_encoders"].get(category)  # Fetch the correct LabelEncoder
    if encoder and hasattr(encoder, "classes_"):
        if value in encoder.classes_:
            return encoder.transform([value])[0]
        else:
            return -1  # Unseen category
    return -1  # Missing encoder

# ✅ Job Description Prediction Section
st.write("### 📌 Predict If a Job is Fake Based on Description")
description = st.text_area("✍️ Enter Job Description Below:")

if st.button("🔮 Predict"):
    if not description.strip():
        st.warning("⚠️ Please enter a job description.")
    else:
        try:
            # BERT Embedding from description
            text_features = get_bert_embedding(description).reshape(1, -1)

            # Sample additional structured features (default/dummy values or based on encoding)
            additional_features = np.array([
                0,  # telecommuting
                1,  # has_company_logo
                0,  # has_questions
                encode_category("Full-time", "employment_type"),
                encode_category("Mid-Senior level", "required_experience"),
                encode_category("Bachelor’s Degree", "required_education"),
                encode_category("Information Technology", "industry"),
                encode_category("Engineering", "function"),
                0,  # department (dummy)
                0,  # salary_range (dummy)
                min(len(description), 1000),  # company_profile (simulated with description length)
                min(len(description), 1000),  # requirements
                min(len(description), 1000)   # benefits
            ]).reshape(1, -1)

            # Combine text and additional features
            full_features = np.hstack((text_features, additional_features))
            full_features_scaled = models["scaler"].transform(np.nan_to_num(full_features))

            # Predict probabilities
            prediction_prob = models["stacking_model"].predict_proba(full_features_scaled)[0]
            fake_confidence, real_confidence = prediction_prob[1] * 100, prediction_prob[0] * 100

            # Prediction Result
            if fake_confidence >= 80:
                st.error(f"⚠️ HIGH RISK: This job is VERY LIKELY FAKE! (Confidence: {fake_confidence:.2f}%)")
            elif fake_confidence >= 60:
                st.warning(f"⚠️ WARNING: This job is PROBABLY FAKE. (Confidence: {fake_confidence:.2f}%)")
            elif real_confidence >= 70:
                st.success(f"✅ SAFE: This job appears LEGIT! (Confidence: {real_confidence:.2f}%)")
            else:
                st.warning(f"⚠️ UNCERTAIN: Fake: {fake_confidence:.2f}%, Real: {real_confidence:.2f}%")

            # Probability Visualization
            st.plotly_chart(px.bar(
                pd.DataFrame({"Category": ["Legit", "Fake"], "Confidence (%)": [real_confidence, fake_confidence]}),
                x="Category", y="Confidence (%)", color="Category",
                text="Confidence (%)", title="Prediction Confidence Levels"
            ))

            # Debugging Info
            st.write("### 🔍 Debugging Info:")
            st.write(f"**Prediction:** {'Fake' if fake_confidence > real_confidence else 'Legit'}")
            st.write(f"**Confidence Levels:** Fake: {fake_confidence:.2f}%, Legit: {real_confidence:.2f}%")
            st.write(f"**Raw Probabilities:** {prediction_prob}")

        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
