import streamlit as st
import pandas as pd
import numpy as np
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
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS with compact styling
st.markdown("""
    <style>
    .main {
        padding: 1rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 2.5em;
        background-color: #4CAF50;
        color: white;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
        box-shadow: 1px 1px 5px rgba(0,0,0,0.1);
        color: #333333; /* Ensuring dark text on light background */
    }
    .insight-box {
        background-color: #ffffff;
        border-left: 5px solid #4CAF50;
        padding: 15px 15px;
        margin: 5px 0;
        border-radius: 5px;
        color: #333333; /* Ensuring dark text on light background */
    }
    .category-high {
        background-color: #e6ffe6;
        border-left: 3px solid #2ecc71;
        color: #0a6d1a; /* Darker green text on light green background */
    }
    .category-medium {
        background-color: #fff5e6;
        border-left: 3px solid #f39c12;
        color: #8a5a00; /* Darker orange text on light orange background */
    }
    .category-low {
        background-color: #ffe6e6;
        border-left: 3px solid #e74c3c;
        color: #9e0500; /* Darker red text on light red background */
    }
    .compact-tabs {
        font-size: 0.9em;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 0.9em;
    }
    div[data-testid="stExpander"] {
        border: none;
        box-shadow: none;
    }
    .compact-text {
        font-size: 0.9em;
        margin: 0;
        padding: 0;
    }
    .stPlotlyChart {
        margin: 0;
        padding: 0;
    }
    /* Make sure cell text in colored cells is readable */
    td {
        color: #333333 !important; /* Forcing dark text in all cells */
    }
    </style>
""", unsafe_allow_html=True)

# Initialize OpenAI client

# Initialize OpenAI client
client = OpenAI(api_key="<key>")

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords')
    nltk.download('wordnet')

download_nltk_data()

# Initialize the lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')

def preprocess(text):
    """Preprocess text by lowercasing, lemmatizing, and removing stopwords."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

@st.cache_data
def load_job_data():
    """Load and prepare the job descriptions data."""
    try:
        job_df = pd.read_csv('/Users/raghav/Desktop/Apexon/JD Optimization/job_descriptions_analysis_output_new_v.csv')
        
        # Initialize all JD types to 'unknown'
        job_df['JD_Type'] = 'unknown'
        
        # Map Java/Python Support JDs - use more flexible matching
        java_python_keywords = ['java', 'python', 'support', 'user_selected_jd']
        
        # Map Principal Software Engineer JDs
        principal_engineer_keywords = ['principal', 'software', 'engineer', 'can']
        
        # Apply more flexible mappings
        for index, row in job_df.iterrows():
            file_name = str(row['File Name']).lower()
            
            # Check for Java/Python developer
            if any(keyword in file_name for keyword in java_python_keywords):
                job_df.at[index, 'JD_Type'] = 'java_developer'
            
            # Check for Principal Engineer
            elif any(keyword in file_name for keyword in principal_engineer_keywords):
                job_df.at[index, 'JD_Type'] = 'principal_engineer'
        
        return job_df
    except Exception as e:
        st.error(f"Error loading job data: {e}")
        return None

@st.cache_data
def load_resume_data(jd_type):
    """Load the appropriate resume data based on the JD type."""
    try:
        if jd_type == 'java_developer':
            resume_df = pd.read_csv('/Users/raghav/Desktop/Apexon/JD Optimization/resumes_analysis_outputJDJavaDeveloper.csv')
        elif jd_type == 'principal_engineer':
            resume_df = pd.read_csv('/Users/raghav/Desktop/Apexon/JD Optimization/resumes_analysis_output_JDPrincipalSoftwareEngineer.csv')
        else:
            # Instead of falling back to All_Resumes.csv, show an error
            st.error(f"No resume data available for job type: {jd_type}")
            return None
        
        return resume_df
    except Exception as e:
        st.error(f"Error loading resume data for {jd_type}: {e}")
        return None

@st.cache_data
def compute_similarity(job_desc, resume_df):
    """Compute similarity scores between job description and resumes."""
    job_desc_processed = preprocess(str(job_desc['Skills'])) + ' ' + preprocess(str(job_desc['Tools']))
    
    # Preprocess resume data
    resume_df['Processed'] = resume_df['Skills'].apply(preprocess) + ' ' + \
                            resume_df['Tools'].apply(preprocess) + ' ' + \
                            resume_df['Certifications'].apply(preprocess)
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    job_desc_matrix = vectorizer.fit_transform([job_desc_processed])
    resume_matrix = vectorizer.transform(resume_df['Processed'])
    
    return cosine_similarity(job_desc_matrix, resume_matrix)[0]

def create_radar_chart(resume, job_desc):
    """Create a radar chart for skill matching visualization."""
    categories = ['Technical Skills', 'Tools Proficiency', 'Certifications']
    
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
        height=250,
        margin=dict(l=20, r=20, t=30, b=20),
        title=None
    )
    return fig

def create_distribution_chart(categorized_resumes):
    """Create a distribution chart showing resume categories."""
    categories = ['High Match', 'Medium Match', 'Low Match']
    counts = [
        len(categorized_resumes['high_matches']),
        len(categorized_resumes['medium_matches']),
        len(categorized_resumes['low_matches'])
    ]
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=counts,
            marker_color=['#2ecc71', '#f39c12', '#e74c3c']
        )
    ])
    
    fig.update_layout(
        title="Match Distribution",
        xaxis_title=None,
        yaxis_title="Count",
        height=200,
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=False
    )
    
    return fig

def categorize_resumes(job_desc, resume_df):
    """Categorize resumes into high, medium, and low matches."""
    similarity_scores = compute_similarity(job_desc, resume_df)
    
    all_resumes = []
    for i, score in enumerate(similarity_scores):
        all_resumes.append({
            'Resume ID': resume_df.iloc[i]['File Name'],
            'Skills': resume_df.iloc[i]['Skills'],
            'Tools': resume_df.iloc[i]['Tools'],
            'Certifications': resume_df.iloc[i]['Certifications'],
            'Score': score
        })
    
    # Sort all resumes by score
    all_resumes.sort(key=lambda x: x['Score'], reverse=True)
    
    # Categorize based on score thresholds
    high_matches = [r for r in all_resumes if r['Score'] >= 0.25]
    medium_matches = [r for r in all_resumes if 0.2 <= r['Score'] < 0.25]
    low_matches = [r for r in all_resumes if r['Score'] < 0.2]
    
    return {
        'top_3': all_resumes[:3],
        'high_matches': high_matches,
        'medium_matches': medium_matches,
        'low_matches': low_matches
    }

def generate_openai_insights(job_desc, resume):
    """Generate AI insights about the resume match."""
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
st.markdown("## 🎯 Smart Resume Ranking System")

# Load the job data
job_df = load_job_data()

if job_df is not None:
    # Create three columns for main layout
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown("### 📋 Job Selection")
        job_desc_file_names = job_df['File Name'].tolist()
        selected_job_desc = st.selectbox('Choose position:', job_desc_file_names, label_visibility="collapsed")
        job_desc = job_df[job_df['File Name'] == selected_job_desc].iloc[0]
        
        # Display the selected JD type for verification (can be removed in production)
        jd_type = job_desc['JD_Type']
        st.markdown(f"**Resume Pool:** {jd_type.replace('_', ' ').title()}")
        
        with st.expander("Job Details", expanded=False):
            st.markdown(f"**Skills:** {job_desc['Skills']}")
            st.markdown(f"**Tools:** {job_desc['Tools']}")
        
        # Load the appropriate resume data based on the selected job
        resume_df = load_resume_data(jd_type)
        
        if resume_df is not None:
            if st.button('🔍 Analyze Resumes'):
                with st.spinner('Analyzing resumes...'):
                    categorized_resumes = categorize_resumes(job_desc, resume_df)
                    st.session_state['analysis_results'] = categorized_resumes
        else:
            st.error(f"Unable to load resume data for {jd_type} job type.")

    if 'analysis_results' in st.session_state:
        categorized_resumes = st.session_state['analysis_results']
        
        with col2:
            st.markdown("### 📊 Overview")
            # Distribution chart
            st.plotly_chart(create_distribution_chart(categorized_resumes), use_container_width=True)
            
            # Top 3 Quick View
            st.markdown("### 🏆 Top Matches")
            for i, resume in enumerate(categorized_resumes['top_3'][:3]):
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="margin:0">#{i + 1} - {resume['Resume ID']}</h4>
                    <p style="margin:0">Match: {resume['Score']:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("### 💫 Detailed Analysis")
            tabs = st.tabs(["#1", "#2", "#3"])
            
            for i, (tab, resume) in enumerate(zip(tabs, categorized_resumes['top_3'])):
                with tab:
                    col_a, col_b = st.columns([1, 1])
                    with col_a:
                        st.markdown(f"**Score:** {resume['Score']:.2%}")
                        radar_chart = create_radar_chart(resume, job_desc)
                        st.plotly_chart(radar_chart, use_container_width=True)
                    
                    with col_b:
                        insights = generate_openai_insights(job_desc, resume)
                        st.markdown(f"""
                        <div class="insight-box compact-text">
                            {insights}
                        </div>
                        """, unsafe_allow_html=True)

        # All Resumes by Category (below the main content)
        st.markdown("---")
        st.markdown("### 📑 All Resumes by Category")
        
        cat_col1, cat_col2, cat_col3 = st.columns(3)
        
        with cat_col1:
            with st.expander(f"High Matches ({len(categorized_resumes['high_matches'])})"):
                for resume in categorized_resumes['high_matches']:
                    st.markdown(f"""
                    <div class="metric-card category-high compact-text">
                        <h4 style="margin:0">{resume['Resume ID']}</h4>
                        <p style="margin:0">Match: {resume['Score']:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with cat_col2:
            with st.expander(f"Medium Matches ({len(categorized_resumes['medium_matches'])})"):
                for resume in categorized_resumes['medium_matches']:
                    st.markdown(f"""
                    <div class="metric-card category-medium compact-text">
                        <h4 style="margin:0">{resume['Resume ID']}</h4>
                        <p style="margin:0">Match: {resume['Score']:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with cat_col3:
            with st.expander(f"Low Matches ({len(categorized_resumes['low_matches'])})"):
                for resume in categorized_resumes['low_matches']:
                    st.markdown(f"""
                    <div class="metric-card category-low compact-text">
                        <h4 style="margin:0">{resume['Resume ID']}</h4>
                        <p style="margin:0">Match: {resume['Score']:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
else:
    st.error("Unable to load job data. Please check your data files and paths.")
