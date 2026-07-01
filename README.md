# 🧠 AI Resume Screener

An AI-powered web app that analyzes resumes and job descriptions to calculate a **match score**, detect **missing skills**, and generate a **visual report** — built using **Streamlit** and **Sentence Transformers**.

---

## 🚀 Features

✅ Ayein your Resume (PDF)  
✅ Paste a Job Description (JD)  
✅ Get an AI-calculated Match Score (%)  
✅ View Skills Found & Missing  
✅ Visual Charts (Pie + Bar graphs)  
✅ Download a PDF Report  
✅ Deployed on [Streamlit Cloud](https://streamlit.io/cloud)

---

## 🧰 Tech Stack

| Category | Tools / Libraries |
|-----------|-------------------|
| **Frontend** | Streamlit |
| **NLP Model** | Sentence Transformers (`all-MiniLM-L6-v2`) |
| **ML Logic** | scikit-learn (Cosine Similarity) |
| **PDF Parsing** | pdfplumber |
| **Charts** | Matplotlib |
| **Report Export** | FPDF |
| **Hosting** | Streamlit Cloud |

---

## 🧩 Project Structure


---

## 🧠 How It Works

1. The app extracts text from the uploaded resume (PDF).  
2. It uses `SentenceTransformer` embeddings to compare with the job description.  
3. Computes cosine similarity → **Match Score (%)**  
4. Extracts keywords to find **skills found** and **skills missing**.  
5. Visualizes the result with **pie and bar charts**.  
6. Generates a **PDF report** for download.

---

## ⚙️ Installation (Local Setup)

```bash
git clone https://github.com/<your-username>/AI_Resume_Screener.git
cd AI_Resume_Screener/app
pip install -r ../requirements.txt
streamlit run main.p
