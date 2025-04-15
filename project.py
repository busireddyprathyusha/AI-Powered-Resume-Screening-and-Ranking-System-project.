import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()

    return cosine_similarities

# Streamlit app UI
st.set_page_config(page_title="AI Resume Screening", page_icon="📄", layout="wide")

st.markdown(
    """
    <style>
    .main-header {
        text-align: center;
        color: #FFD700;
        font-size: 3rem;
    }
    .sub-header {
        text-align: center;
        font-size: 1.2rem;
        color: #DDDDDD;
    }
    .home-container {
        text-align: center;
        margin-top: 50px;
    }
    .graph-section {
        padding: 20px;
        background-color: #1e1e1e;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.title("Navigation")
st.sidebar.markdown("Use the options below to navigate:")
option = st.sidebar.radio("Go to", ["🏠 Home", "📂 Upload Resumes", "📊 Results"])

if option == "🏠 Home":
    st.markdown("""
    <div class='home-container'>
    <h1 class='main-header'>💼 AI Resume Screening & Candidate Ranking</h1>
    <p class='sub-header'>Upload resumes and get them ranked based on the job description.</p>
    <h3>🚀 Key Features</h3>
    - ✅ Upload multiple resumes<br>
    - ✅ Compare them with job descriptions<br>
    - ✅ Download ranked results<br>
    - ✅ Visualize insights through graphs
    </div>
    """, unsafe_allow_html=True)

elif option == "📂 Upload Resumes":
    job_description = st.text_area("📝 Enter Job Description", height=150)
    uploaded_files = st.file_uploader("📂 Upload PDF Resumes", type=["pdf"], accept_multiple_files=True)

    if uploaded_files and job_description:
        st.success("✅ Files uploaded successfully! Navigate to Results.")
        st.session_state.uploaded_files = uploaded_files
        st.session_state.job_description = job_description

elif option == "📊 Results":
    if 'uploaded_files' in st.session_state and 'job_description' in st.session_state:
        resumes = [extract_text_from_pdf(file) for file in st.session_state.uploaded_files]
        scores = rank_resumes(st.session_state.job_description, resumes)

        results = pd.DataFrame({
            "Resume": [file.name for file in st.session_state.uploaded_files],
            "Score": scores
        }).sort_values(by="Score", ascending=False)

        # Bar Graph with Clear Insights
        st.markdown("<div class='graph-section'><h3>📊 Match Score Distribution</h3></div>", unsafe_allow_html=True)
        fig = go.Figure(
            data=[
                go.Bar(
                    x=results['Score'] * 100,
                    y=results['Resume'],
                    orientation='h',
                    marker_color='lightseagreen',
                    text=[f"{round(score*100,2)}%" for score in results['Score']],
                    textposition='auto'
                )
            ]
        )
        fig.update_layout(title_text="Match Score of Each Resume", xaxis_title="Match Score (%)", yaxis_title="Resumes")
        st.plotly_chart(fig)

        # Pie Chart with Clear Insights
        st.markdown("<div class='graph-section'><h3>🎯 Match Score Proportions</h3></div>", unsafe_allow_html=True)
        fig1 = px.pie(
            results,
            values='Score',
            names='Resume',
            title='Proportion of Each Resume Match',
            color_discrete_sequence=px.colors.sequential.Teal)
        st.plotly_chart(fig1)

        # Heatmap with Clear Insights
        st.markdown("<div class='graph-section'><h3>📊 Heatmap Correlation</h3></div>", unsafe_allow_html=True)
        fig2 = go.Figure(data=go.Heatmap(
            z=[results['Score'] * 100],
            x=results['Resume'],
            colorscale='Blues'
        ))
        fig2.update_layout(title_text="Heatmap of Resume Match Scores", xaxis_title="Resumes", yaxis_title="Match Score (%)")
        st.plotly_chart(fig2)

        # Top Candidates
        st.subheader("🏆 Top Candidates")
        for index, row in results.head(5).iterrows():
            st.markdown(f"""
            <div class='graph-section'>
                <h4>📜 {row['Resume']}</h4>
                <p><b>Match Score:</b> {round(row['Score'] * 100, 2)}%</p>
                <div style='height: 8px; background-color: #00c853; width: {round(row['Score'] * 100, 2)}%;'></div>
            </div>
            """, unsafe_allow_html=True)

        # Download CSV
        st.download_button(
            label="📥 Download Ranking CSV",
            data=results.to_csv(index=False).encode('utf-8'),
            file_name="ranked_resumes.csv",
            mime='text/csv'
        )
    else:
        st.warning("No files uploaded or job description provided.")
