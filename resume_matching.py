import streamlit as st
import PyPDF2
import torch
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

# Load BERT model for embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    """Extracts text from an uploaded PDF file."""
    text = ""
    reader = PyPDF2.PdfReader(uploaded_file)
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

# Function to calculate similarity scores
def compute_similarity(job_desc, resumes):
    job_embedding = model.encode(job_desc, convert_to_tensor=True)
    resume_embeddings = [model.encode(resume, convert_to_tensor=True) for resume in resumes]
    
    similarity_scores = [util.pytorch_cos_sim(job_embedding, res_emb)[0].item() for res_emb in resume_embeddings]
    return similarity_scores

# Function to rank resumes based on similarity to job description
def rank_resumes(resume_texts, job_description):
    """Ranks resumes by similarity to the job description using BERT embeddings."""
    job_embedding = model.encode(job_description, convert_to_tensor=True)

    scores = []
    for resume_text in resume_texts:
        resume_embedding = model.encode(resume_text, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(job_embedding, resume_embedding)
        scores.append(similarity.item())

    ranked_resumes = sorted(zip(resume_texts, scores), key=lambda x: x[1], reverse=True)
    return ranked_resumes


# Streamlit UI
st.title("📄 Resume Matching & Evaluation")

job_desc = st.text_area("Enter Job Description:", "")
uploaded_files = st.file_uploader("Upload Resume PDFs", type=["pdf"], accept_multiple_files=True)

if st.button("Match Resumes"):
    if not job_desc or not uploaded_files:
        st.warning("Please enter a job description and upload resumes.")
    else:
        resumes_text = [extract_text_from_pdf(file) for file in uploaded_files]
        ranked_resumes = rank_resumes(resumes_text, job_desc)

        # Extract rankings
        ranked_texts, similarity_scores = zip(*ranked_resumes)


        # Display Results
        results_df = pd.DataFrame({
            "Resume": [file.name for file in uploaded_files],
            "Similarity Score": similarity_scores
        }).sort_values(by="Similarity Score", ascending=False)

        st.write("### 🔹 Resume Ranking:")
        st.dataframe(results_df)

        st.success("Resume ranking  completed!")
