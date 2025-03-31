import os
import numpy as np
import pandas as pd
import tempfile
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from docx import Document
from utils.text_processing import extract_skills, preprocess_text

class ResumeAnalyzer:
    """Analyze and rank resumes based on job descriptions"""
    
    def __init__(self):
        """Initialize the ResumeAnalyzer"""
        self.vectorizer = TfidfVectorizer()
    
    def compute_similarity(self, job_desc, resume_df):
        """
        Compute enhanced similarity scores between job description and resumes
        
        Args:
            job_desc (dict): Job description with Skills and Tools fields
            resume_df (DataFrame): DataFrame containing resume data
            
        Returns:
            numpy.ndarray: Array of similarity scores
        """
        # Extract skills from job description
        job_skills = extract_skills(str(job_desc['Skills']) + ' ' + str(job_desc['Tools']))
        
        similarity_scores = []
        for _, resume in resume_df.iterrows():
            # Extract skills from resume
            resume_skills = extract_skills(str(resume['Skills']) + ' ' + str(resume['Tools']))
            
            # Calculate skill matches for each category
            category_scores = []
            for category in job_skills:
                job_set = set(job_skills[category])
                resume_set = set(resume_skills[category])
                if job_set:
                    match_ratio = len(resume_set.intersection(job_set)) / len(job_set)
                    category_scores.append(match_ratio)
                else:
                    category_scores.append(0)
            
            # Calculate weighted skill score
            skill_score = np.mean(category_scores) if category_scores else 0
            
            # Calculate text similarity
            job_text = preprocess_text(str(job_desc['Skills']) + ' ' + str(job_desc['Tools']))
            resume_text = preprocess_text(
                str(resume['Skills']) + ' ' + 
                str(resume['Tools']) + ' ' + 
                str(resume['Certifications'])
            )
            
            try:
                tfidf_matrix = self.vectorizer.fit_transform([job_text, resume_text])
                text_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            except:
                text_similarity = 0
            
            # Combine scores (70% skill match, 30% text similarity)
            final_score = (0.7 * skill_score) + (0.3 * text_similarity)
            similarity_scores.append(final_score)
        
        return np.array(similarity_scores)
    
    def categorize_resumes(self, job_desc, resume_df):
        """
        Categorize resumes into high, medium, and low matches
        
        Args:
            job_desc (dict): Job description with Skills and Tools fields
            resume_df (DataFrame): DataFrame containing resume data
            
        Returns:
            dict: Dictionary with categorized resumes
        """
        similarity_scores = self.compute_similarity(job_desc, resume_df)
        
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
    
    def load_resume_data(self, jd_type=None):
        """
        Load resume data by letting the user select from available files
        
        Args:
            jd_type (str, optional): Type of job description (used only for display)
            
        Returns:
            DataFrame: DataFrame containing resume data
        """
        try:
            # Check for resume directory with multiple possible spellings
            directories_to_check = [
                "Exctracted Resumes",  # Match your actual directory name first
                "Extracted Resumes", 
                "ExtractedResumes", 
                "Resumes"
            ]
            
            extracted_dir = None
            for dir_name in directories_to_check:
                if os.path.exists(dir_name):
                    extracted_dir = dir_name
                    break
            
            # Get all CSV files either from the directory or current directory
            if extracted_dir:
                st.info(f"Using resume directory: {extracted_dir}")
                resume_files = [f for f in os.listdir(extracted_dir) if f.endswith('.csv')]
            else:
                # Look for CSV files in current directory if no resume folder exists
                resume_files = [f for f in os.listdir() if f.endswith('.csv')]
            
            if not resume_files:
                st.warning(f"No resume CSV files found. Please add resume files to continue.")
                return None
            
            # Let user select a file from dropdown
            selected_file = st.selectbox(
                "Select Resume Data File:",
                options=resume_files,
                help="Choose a CSV file containing resume data"
            )
            
            # Determine the full path to the selected file
            if extracted_dir and os.path.exists(os.path.join(extracted_dir, selected_file)):
                file_path = os.path.join(extracted_dir, selected_file)
            else:
                file_path = selected_file
                
            # Read the selected CSV file
            resume_df = pd.read_csv(file_path)
            return resume_df
        
        except Exception as e:
            st.error(f"Error loading resume data: {e}")
            
            # Create and return sample data instead
            st.info("Using sample resume data instead")
            sample_resume_data = {
                'File Name': ['Resume_1', 'Resume_2', 'Resume_3', 'Resume_4', 'Resume_5'],
                'Skills': [
                    'Python, Java, Data Analysis, Machine Learning', 
                    'Java, Python, SQL, REST API',
                    'C#, .NET, Azure, Cloud Computing',
                    'Java, Spring, Hibernate, SQL, REST',
                    'Python, ML, AI, Deep Learning, SQL'
                ],
                'Tools': [
                    'TensorFlow, Scikit-learn, Docker, Git', 
                    'IntelliJ, Eclipse, Git, Maven',
                    'Visual Studio, Git, Azure DevOps',
                    'Jenkins, Maven, Docker, Kubernetes',
                    'Pandas, NumPy, Jupyter, Keras'
                ],
                'Certifications': [
                    'AWS Machine Learning Specialty', 
                    'Oracle Java Professional',
                    'Microsoft Azure Developer',
                    'AWS Developer Associate',
                    'Google Professional Data Engineer'
                ]
            }
            return pd.DataFrame(sample_resume_data)
    
    def analyze_uploaded_resume(self, uploaded_file):
        """
        Analyze a user-uploaded resume (.docx) and return the extracted information.
        
        Args:
            uploaded_file (UploadedFile): The uploaded resume file
            
        Returns:
            dict: Dictionary with extracted resume details
        """
        # Only process .docx files
        if not uploaded_file.name.endswith(".docx"):
            raise ValueError(f"Unsupported file format for {uploaded_file.name}. Only .docx files are supported.")
        
        # Write the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        try:
            # Extract text from the document
            doc = Document(tmp_path)
            resume_text = "\n".join([para.text for para in doc.paragraphs])
            
            # Basic extraction - in a real implementation, you would use NLP or an LLM
            # to extract these details more accurately
            skills = extract_skills(resume_text)
            skills_str = ", ".join([item for sublist in skills.values() for item in sublist])
            
            # Just as a simple example - detecting tools is more complex in reality
            tools_keywords = ['git', 'docker', 'kubernetes', 'jenkins', 'jira', 
                             'confluence', 'aws', 'azure', 'vs code', 'intellij']
            detected_tools = []
            for tool in tools_keywords:
                if tool.lower() in resume_text.lower():
                    detected_tools.append(tool)
            
            # Similarly, certifications would need better extraction
            cert_keywords = ['certified', 'certification', 'certificate', 'aws', 'azure', 
                           'google', 'professional', 'associate', 'expert']
            has_cert = any(kw in resume_text.lower() for kw in cert_keywords)
            
            return {
                'File Name': uploaded_file.name,
                'Skills': skills_str or "General programming, problem-solving",
                'Tools': ", ".join(detected_tools) or "Standard development tools",
                'Certifications': "Certifications detected" if has_cert else "None specified"
            }
        except Exception as e:
            print(f"Error in analyze_uploaded_resume for {uploaded_file.name}: {e}")
            return None
        finally:
            # Always remove the temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def process_resume_pool(self, uploaded_files):
        """
        Process a batch of uploaded resume files and return a DataFrame
        
        Args:
            uploaded_files (list): List of uploaded resume files
            
        Returns:
            DataFrame: DataFrame containing processed resume data
        """
        processed_resumes = []
        for uploaded_file in uploaded_files:
            try:
                if uploaded_file.name.endswith(".docx"):
                    resume_data = self.analyze_uploaded_resume(uploaded_file)
                    if resume_data is not None:
                        processed_resumes.append(resume_data)
                else:
                    print(f"Skipping {uploaded_file.name} - not a .docx file")
            except Exception as e:
                print(f"Error processing {uploaded_file.name}: {e}")
        
        if processed_resumes:
            return pd.DataFrame(processed_resumes)
        return None