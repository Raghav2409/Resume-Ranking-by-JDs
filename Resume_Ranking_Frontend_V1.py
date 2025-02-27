import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from openai import OpenAI
import plotly.graph_objects as go
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Resume Ranking System",
    page_icon="📄",
    layout="wide"
)

# Custom CSS to inject
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .insight-box {
        background-color: #ffffff;
        border-left: 5px solid #4CAF50;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize OpenAI client
client = OpenAI(api_key="OpenAI Key")

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords')
    nltk.download('wordnet')

download_nltk_data()

# Initialize the lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')

# Preprocessing function (same as before)
def preprocess(text):
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Load data function (same as before)
@st.cache_data
def load_data():
    job_df = pd.read_csv('/Users/raghav/Desktop/Apexon/JD Optimization/job_descriptions_analysis_output v2.csv')
    resume_df = pd.read_csv('/Users/raghav/Desktop/Apexon/JD Optimization/resumes_analysis_output.csv')
    
    job_df['File Name'] = job_df['File Name'].replace({
        'JobDescriptionJavaPythonSupport.txt': 'Java/Python Support',
        'DataAnalyticsAIMLJD (1).txt': 'Data Analytics/AIML',
        'User_Selected_JD.txt': 'Enhanced Java/Python Support JD'
    })
    
    return job_df, resume_df

# TF-IDF and similarity computation (same as before)
@st.cache_data
def compute_similarity(job_desc, resume_df):
    job_desc_processed = preprocess(str(job_desc['Skills'])) + ' ' + preprocess(str(job_desc['Tools']))
    resume_df['Processed'] = resume_df['Skills'].apply(lambda x: preprocess(str(x))) + ' ' + \
                            resume_df['Tools'].apply(lambda x: preprocess(str(x))) + ' ' + \
                            resume_df['Certifications'].apply(lambda x: preprocess(str(x)))
    
    vectorizer = TfidfVectorizer()
    job_desc_matrix = vectorizer.fit_transform([job_desc_processed])
    resume_matrix = vectorizer.transform(resume_df['Processed'])

    return cosine_similarity(job_desc_matrix, resume_matrix)[0]

def create_radar_chart(resume, job_desc):
    # Create normalized scores for different aspects
    categories = ['Technical Skills', 'Tools Proficiency', 'Certifications']
    
    # Simple scoring based on word overlap
    resume_skills = set(str(resume['Skills']).lower().split())
    resume_tools = set(str(resume['Tools']).lower().split())
    resume_certs = set(str(resume['Certifications']).lower().split())
    
    job_skills = set(str(job_desc['Skills']).lower().split())
    job_tools = set(str(job_desc['Tools']).lower().split())
    
    scores = [
        len(resume_skills & job_skills) / max(len(job_skills), 1),
        len(resume_tools & job_tools) / max(len(job_tools), 1),
        len(resume_certs) / 10  # Normalize certification count
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        name='Match Score'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title="Profile Match Analysis"
    )
    return fig

# Rank resumes function with visualization
def rank_resumes(job_desc, resume_df):
    similarity_scores = compute_similarity(job_desc, resume_df)
    
    resume_scores = []
    for i, score in enumerate(similarity_scores):
        resume_scores.append({
            'Resume ID': resume_df.iloc[i]['File Name'],
            'Skills': resume_df.iloc[i]['Skills'],
            'Tools': resume_df.iloc[i]['Tools'],
            'Certifications': resume_df.iloc[i]['Certifications'],
            'Score': score
        })

    return sorted(resume_scores, key=lambda x: x['Score'], reverse=True)[:3]

# Generate insights function (modified for better formatting)
def generate_openai_insights(job_desc, resume):
    prompt = f"""
    As an expert in talent evaluation, analyze this job description and resume:
    Job Description:
    Skills: {job_desc['Skills']}
    Tools: {job_desc['Tools']}
    
    Resume:
    Skills: {resume['Skills']}
    Tools: {resume['Tools']}
    Certifications: {resume['Certifications']}
    
    Provide:
    1. A concise paragraph on why the candidate is a good fit
    2. 3 key strengths of this candidate's profile (in bullet points)
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert in HR and talent evaluation."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating insights: {e}"

# Main UI
st.title('🎯 Smart Resume Ranking System')
st.markdown('---')

# Load the data
job_df, resume_df = load_data()

# Create two columns for layout
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 📋 Select Job Description")
    job_desc_file_names = job_df['File Name'].tolist()
    selected_job_desc = st.selectbox('Choose a position:', job_desc_file_names)
    job_desc = job_df[job_df['File Name'] == selected_job_desc].iloc[0]
    
    with st.expander("View Job Details", expanded=True):
        st.markdown(f"**Required Skills:**\n{job_desc['Skills']}")
        st.markdown(f"**Required Tools:**\n{job_desc['Tools']}")
    
    if st.button('🔍 Find Top Matches'):
        ranked_resumes = rank_resumes(job_desc, resume_df)
        
        with col2:
            st.markdown("### 🏆 Top Matching Candidates")
            
            for i, resume in enumerate(ranked_resumes):
                with st.container():
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>#{i + 1} - {resume['Resume ID']}</h4>
                        <p>Match Score: {resume['Score']:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create tabs for different views
                    tab1, tab2, tab3 = st.tabs(["📊 Match Analysis", "💡 AI Insights", "📄 Details"])
                    
                    with tab1:
                        radar_chart = create_radar_chart(resume, job_desc)
                        st.plotly_chart(radar_chart, use_container_width=True)
                    
                    with tab2:
                        insights = generate_openai_insights(job_desc, resume)
                        st.markdown(f"""
                        <div class="insight-box">
                            {insights}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with tab3:
                        st.markdown("**Skills:**")
                        st.write(resume['Skills'])
                        st.markdown("**Tools:**")
                        st.write(resume['Tools'])
                        st.markdown("**Certifications:**")
                        st.write(resume['Certifications'])
                    
                    st.markdown("---")
