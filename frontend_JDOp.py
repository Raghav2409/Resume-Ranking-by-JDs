import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from openai import OpenAI # Import OpenAI library

# Set up OpenAI API key
client = OpenAI(api_key="open-ai key")
# Download NLTK data (required for stopwords and lemmatizer)
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')

# Preprocessing function
def preprocess(text):
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Load job descriptions and resumes data
@st.cache_data
def load_data():
    job_df = pd.read_csv('/Users/raghav/Desktop/Apexon/JD Optimization/job_descriptions_analysis_output.csv')
    resume_df = pd.read_csv('/Users/raghav/Desktop/Apexon/JD Optimization/resumes_analysis_output.csv')
    
    # Rename job descriptions in the file to friendly names
    job_df['File Name'] = job_df['File Name'].replace({
        'JobDescriptionJavaPythonSupport.txt': 'Java/Python Support',
        'DataAnalyticsAIMLJD (1).txt': 'Data Analytics/AIML'
    })
    
    return job_df, resume_df

# TF-IDF Vectorizer and cosine similarity function
@st.cache_data
def compute_similarity(job_desc, resume_df):
    job_desc_processed = preprocess(str(job_desc['Skills'])) + ' ' + preprocess(str(job_desc['Tools']))
    
    # Preprocess resume data
    resume_df['Processed'] = resume_df['Skills'].apply(lambda x: preprocess(str(x))) + ' ' + resume_df['Tools'].apply(lambda x: preprocess(str(x))) + ' ' + resume_df['Certifications'].apply(lambda x: preprocess(str(x)))
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    job_desc_matrix = vectorizer.fit_transform([job_desc_processed])
    resume_matrix = vectorizer.transform(resume_df['Processed'])

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(job_desc_matrix, resume_matrix)[0]
    return similarity_scores

# Rank resumes based on similarity scores
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

    # Sort by score and return top 3
    ranked_resumes = sorted(resume_scores, key=lambda x: x['Score'], reverse=True)[:3]
    return ranked_resumes

# Generate insights using OpenAI
def generate_openai_insights(job_desc, resume):
    prompt = f"""
    You are an expert in talent evaluation. Analyze the following job description and resume:
    - Job Description:
        Skills: {job_desc['Skills']}
        Tools: {job_desc['Tools']}
    - Resume:
        Skills: {resume['Skills']}
        Tools: {resume['Tools']}
        Certifications: {resume['Certifications']}
    
    Provide:
    1. Why the candidate is a good fit for the job.
    2. What are the strengths of this candidate's profile?
    """

    try:
        response = client.chat.completions.create(  # Updated API call
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert in HR and talent evaluation."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()  # Updated response parsing
    except Exception as e:
        return f"Error generating insights: {e}"
        

# Streamlit UI
st.title('Resume Ranking for Job Descriptions')

# Load the data
job_df, resume_df = load_data()

# Select Job Description from the dropdown
job_desc_file_names = job_df['File Name'].tolist()
selected_job_desc = st.selectbox('Choose a Job Description:', job_desc_file_names)

# Filter the selected job description
job_desc = job_df[job_df['File Name'] == selected_job_desc].iloc[0]

# Display the contents of the selected job description
st.write(f"### Job Description: {selected_job_desc}")
st.write(f"**Skills**: {job_desc['Skills']}")
st.write(f"**Tools**: {job_desc['Tools']}")

# Rank resumes for the selected job description
if st.button('Rank Resumes'):
    ranked_resumes = rank_resumes(job_desc, resume_df)
    
    # Display results
    st.write(f"### Top 3 Ranked Resumes for {selected_job_desc}")
    for i, resume in enumerate(ranked_resumes):
        st.write(f"**{i + 1}. Resume ID**: {resume['Resume ID']}")
        st.write(f"**Match Score**: {resume['Score']:.4f}")
        
        # Generate insights using OpenAI
        insights = generate_openai_insights(job_desc, resume)
        st.write(insights)
        st.write("---")
