# ==========================
# AI Resume Screener v2.0
# Developed by Vedant Padwal
# ==========================

import streamlit as st
import pdfplumber
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import numpy as np
import os
from fpdf import FPDF
import datetime
import re
from wordcloud import WordCloud

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AI Resume Screener 🧠",
    page_icon="🧠",
    layout="wide"
)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://static.streamlit.io/examples/dice.jpg", width=80)
    st.title("⚙️ About this App")
    st.write("""
    **AI Resume Screener** uses NLP to match resumes against job descriptions,
    highlight skill gaps, and generate detailed AI-based insights.
    """)
    st.markdown("---")
    st.write("👨‍💻 **Developer:** Tanay & Samarth")
    st.write("🔗 [GitHub Repo](https://github.com/vedantpadwal/AI_Resume_Screener)")
    st.caption("Version 2.0 | Powered by Streamlit + Transformers")

# --- HEADER ---
st.title("🧠 AI Resume Screener")
st.markdown("Upload your **Resume (PDF)** and a **Job Description** to find how well they match.")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# --- SKILL LIST ---
ai_skills = [
    "python", "machine learning", "deep learning", "tensorflow", "pytorch", "nlp",
    "data analysis", "sql", "computer vision", "cnn", "transformers", "hugging face",
    "pandas", "numpy", "data preprocessing", "feature engineering", "model deployment",
    "aws", "sagemaker", "docker", "kubernetes", "mlflow", "airflow", "scikit-learn",
    "java", "c++", "big data", "data visualization", "fastapi", "flask", "opencv"
]

# --- INPUT SECTION ---
uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type="pdf")
job_desc = st.text_area("💼 Paste Job Description", height=200)

# --- PDF TEXT EXTRACTION ---
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

# --- MAIN APP LOGIC ---
if uploaded_file:
    st.success("✅ Resume uploaded successfully!")

    resume_text = extract_text_from_pdf(uploaded_file)
    st.subheader("📋 Extracted Resume Text:")
    st.text_area("Resume Content", resume_text, height=250)

    if job_desc:
        st.subheader("🧠 AI Analysis in Progress...")
        with st.spinner("Analyzing resume vs job description..."):
            resume_vec = model.encode([resume_text])[0].reshape(1, -1)
            jd_vec = model.encode([job_desc])[0].reshape(1, -1)
            score = cosine_similarity(resume_vec, jd_vec)[0][0] * 100

            # SKILL EXTRACTION
            resume_lower = resume_text.lower()
            jd_lower = job_desc.lower()
            found_skills = [s for s in ai_skills if s in resume_lower]
            required_skills = [s for s in ai_skills if s in jd_lower]
            missing_skills = [s for s in required_skills if s not in found_skills]

        # --- DISPLAY METRICS ---
        col1, col2 = st.columns(2)
        with col1:
            st.metric("✨ Match Score", f"{score:.2f}%")
            if score > 80:
                st.success("✅ Excellent Fit!")
            elif score > 50:
                st.warning("⚠️ Moderate Fit — could be improved.")
            else:
                st.error("❌ Weak Fit — needs improvement.")

        with col2:
            fig, ax = plt.subplots(figsize=(3.8, 3.8))
            wedges, texts, autotexts = ax.pie(
                [score, 100 - score],
                labels=["Match", "Gap"],
                autopct='%1.1f%%',
                startangle=90,
                colors=["#3E8E7E", "#E0E0E0"]
            )
            for w in wedges:
                w.set_edgecolor('white')
            ax.set_title("📊 Resume–JD Match Ratio", fontsize=12, fontweight="bold")
            st.pyplot(fig)

        # --- SKILL ANALYSIS ---
        st.markdown("---")
        st.subheader("🔍 Skill Analysis")

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**✅ Skills Found in Resume:**")
            if found_skills:
                st.success(", ".join(found_skills))
            else:
                st.info("_No key AI/ML skills detected in resume._")

        with col4:
            st.markdown("**❌ Missing Skills (from JD):**")
            if missing_skills:
                st.error(", ".join(missing_skills))
            else:
                st.success("All key skills are present! 🚀")

        # --- SKILL SUGGESTIONS ---
        if missing_skills:
            st.subheader("💡 Suggestions to Improve Fit")
            st.write("You could improve your profile by learning or highlighting:")
            for skill in missing_skills:
                st.markdown(f"- {skill.title()}")
        else:
            st.success("Your resume already covers all key job requirements!")

        # --- WORD CLOUD ---
        st.markdown("### ☁️ Skill Word Cloud")
        text_combined = resume_text + " " + job_desc
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_combined)
        fig_wc, ax_wc = plt.subplots(figsize=(8, 4))
        ax_wc.imshow(wordcloud, interpolation='bilinear')
        ax_wc.axis("off")
        st.pyplot(fig_wc)

        # --- SKILL COVERAGE CHART ---
        st.markdown("### 📊 Skill Coverage Overview")
        if required_skills:
            found_count = len(found_skills)
            missing_count = len(missing_skills)

            fig2, ax2 = plt.subplots(figsize=(4.5, 3.2))
            bars = ax2.bar(
                ["Skills Found", "Missing Skills"],
                [found_count, missing_count],
                color=["#4A90E2", "#FF6B6B"]
            )
            ax2.set_ylabel("Count")
            ax2.set_title("Skill Coverage", fontsize=12, fontweight="bold")
            ax2.grid(axis="y", linestyle="--", alpha=0.5)
            for bar in bars:
                yval = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.1, int(yval),
                         ha='center', va='bottom', fontsize=9)
            st.pyplot(fig2)
        else:
            st.info("No clear skills detected in job description.")

        # --- SAFE SAVE REPORT ---
        data_dir = "../data"
        os.makedirs(data_dir, exist_ok=True)
        report_path = os.path.join(data_dir, "skill_report_visual.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Match Score: {score:.2f}%\n")
            f.write(f"Skills Found: {found_skills}\n")
            f.write(f"Missing Skills: {missing_skills}\n")

        # --- PDF REPORT ---
        st.markdown("### 📄 Download Your AI Screening Report")

        if st.button("Generate Report PDF"):
            def clean_text(text):
                return re.sub(r'[^\x00-\x7F]+', ' ', text)

            report = FPDF()
            report.add_page()
            report.set_font("Arial", 'B', 16)
            report.cell(0, 10, "AI Resume Screener Report", ln=True, align="C")
            report.ln(10)

            clean_score = f"{score:.2f}%"
            clean_found = clean_text(", ".join(found_skills)) if found_skills else "None"
            clean_missing = clean_text(", ".join(missing_skills)) if missing_skills else "None"
            clean_jd = clean_text(job_desc[:500])

            report.set_font("Arial", size=12)
            report.cell(0, 10, f"Match Score: {clean_score}", ln=True)
            report.ln(5)
            report.cell(0, 10, f"Skills Found: {clean_found}", ln=True)
            report.ln(5)
            report.cell(0, 10, f"Missing Skills: {clean_missing}", ln=True)
            report.ln(10)
            report.multi_cell(0, 8, f"Job Description Summary:\n{clean_jd}...", align="L")
            report.ln(10)
            report.cell(0, 10, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)

            pdf_path = "../data/AI_Screener_Report.pdf"
            report.output(pdf_path)

            st.success("✅ Report generated successfully!")
            with open(pdf_path, "rb") as file:
                st.download_button(
                    label="⬇️ Download Report PDF",
                    data=file,
                    file_name="AI_Screener_Report.pdf",
                    mime="application/pdf"
                )

        # --- TEXT SUMMARY DOWNLOAD ---
        summary_content = f"""
AI Resume Screener Report
-------------------------
Match Score: {score:.2f}%
Skills Found: {', '.join(found_skills)}
Missing Skills: {', '.join(missing_skills)}
"""
        st.download_button(
            label="📑 Download Text Summary",
            data=summary_content,
            file_name="AI_Screener_Summary.txt",
            mime="text/plain"
        )

    else:
        st.warning("⚠️ Please paste a Job Description before analysis.")
else:
    st.info("📄 Please upload your resume to start analysis.")

# --- FOOTER ---
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: grey;'>"
    "Built with ❤️ by <b>Tanay & Samarth</b> | "
    "Powered by <a href='https://streamlit.io' target='_blank'>Streamlit</a>"
    "</p>",
    unsafe_allow_html=True
)
