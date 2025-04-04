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
        # Get the base directory (where your app is running)
        self.base_dir = os.getcwd()
        # Define the specific path to the Extracted Resumes folder
        self.resume_dir = os.path.join(self.base_dir, "Data/Extracted Resumes")
    
    def compute_similarity(self, job_desc, resume_df):
        """
        Compute enhanced similarity scores between job description and resumes
        
        Args:
            job_desc (dict): Job description with Skills and Tools fields
            resume_df (DataFrame): DataFrame containing resume data
            
        Returns:
            numpy.ndarray: Array of similarity scores
        """
        # Check if job_desc is valid
        if not isinstance(job_desc, (dict, pd.Series)) or 'Skills' not in job_desc:
            st.warning("Invalid job description format. Using empty skills for comparison.")
            job_skills = {'programming_languages': [], 'frameworks': [], 'databases': [], 'cloud': [], 'tools': []}
        else:
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
            job_text = preprocess_text(str(job_desc.get('Skills', '')) + ' ' + str(job_desc.get('Tools', '')))
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
        # Check if inputs are valid
        if resume_df is None or len(resume_df) == 0:
            st.warning("No resume data available for analysis")
            empty_result = {
                'top_3': [],
                'high_matches': [],
                'medium_matches': [],
                'low_matches': []
            }
            return empty_result
            
        # Compute similarity scores
        try:
            similarity_scores = self.compute_similarity(job_desc, resume_df)
        except Exception as e:
            st.error(f"Error computing similarity: {str(e)}")
            # Return empty results in case of error
            empty_result = {
                'top_3': [],
                'high_matches': [],
                'medium_matches': [],
                'low_matches': []
            }
            return empty_result
        
        all_resumes = []
        for i, score in enumerate(similarity_scores):
            # Make sure we don't go out of bounds
            if i < len(resume_df):
                resume_row = resume_df.iloc[i]
                all_resumes.append({
                    'Resume ID': resume_row.get('File Name', f"Resume_{i+1}"),
                    'Skills': resume_row.get('Skills', ''),
                    'Tools': resume_row.get('Tools', ''),
                    'Certifications': resume_row.get('Certifications', ''),
                    'Score': score
                })
        
        # Sort all resumes by score
        all_resumes.sort(key=lambda x: x['Score'], reverse=True)
        
        # Categorize based on score thresholds
        high_matches = [r for r in all_resumes if r['Score'] >= 0.25]
        medium_matches = [r for r in all_resumes if 0.2 <= r['Score'] < 0.25]
        low_matches = [r for r in all_resumes if r['Score'] < 0.2]
        
        return {
            'top_3': all_resumes[:3] if len(all_resumes) >= 3 else all_resumes,
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
            # Check if the specific Extracted Resumes directory exists
            if os.path.exists(self.resume_dir) and os.path.isdir(self.resume_dir):
                st.info(f"Using resume directory: Data/Extracted Resumes")
                resume_files = [f for f in os.listdir(self.resume_dir) if f.endswith('.csv')]
            else:
                # Fallback to looking in the Data directory
                data_dir = os.path.join(self.base_dir, "Data")
                st.warning("'Data/Extracted Resumes' directory not found. Looking for resume files in the Data directory.")
                resume_files = []
                for root, _, files in os.walk(data_dir):
                    for file in files:
                        if file.endswith('.csv') and ('resume' in file.lower() or 'analysis' in file.lower()):
                            resume_files.append(os.path.join(root, file))
                
                # If found resume files, update resume_dir
                if resume_files:
                    common_dir = os.path.commonpath([os.path.dirname(f) for f in resume_files])
                    self.resume_dir = common_dir
            
            # If still no files found, try specific file names based on jd_type
            if not resume_files and jd_type:
                default_files = {
                    "java_developer": ["resumes_analysis_outputJDJavaDeveloper.csv", "java_resumes.csv"],
                    "data_engineer": ["resumes_analysis_output_JDPrincipalSoftwareEngineer.csv", "data_resumes.csv"],
                    "general": ["resumes_analysis_output.csv", "resumes.csv"]
                }
                
                default_file_list = default_files.get(jd_type, ["resumes_analysis_output.csv"])
                
                for file_name in default_file_list:
                    # Try in Data/Extracted Resumes
                    potential_path = os.path.join(self.base_dir, "Data", "Extracted Resumes", file_name)
                    if os.path.exists(potential_path):
                        resume_files = [potential_path]
                        self.resume_dir = os.path.join(self.base_dir, "Data", "Extracted Resumes")
                        st.info(f"Using resume file for {jd_type}: {file_name}")
                        break
                    
                    # Try in base/Data directory recursively
                    for root, _, files in os.walk(os.path.join(self.base_dir, "Data")):
                        if file_name in files:
                            resume_files = [os.path.join(root, file_name)]
                            self.resume_dir = root
                            st.info(f"Found resume file for {jd_type}: {file_name}")
                            break
            
            if not resume_files:
                st.warning("No resume CSV files found. Using sample data.")
                return self.create_sample_resume_df()
            
            # If resume files are full paths, get basenames for display
            resume_files_display = [os.path.basename(f) if os.path.isabs(f) else f for f in resume_files]
            
            # Let user select a file from dropdown
            selected_file_display = st.selectbox(
                "Select Resume Data File:",
                options=resume_files_display,
                help="Choose a CSV file containing resume data"
            )
            
            # Get the full path of the selected file
            selected_file = next((f for f in resume_files if os.path.basename(f) == selected_file_display or f == selected_file_display), None)
            
            if not selected_file:
                st.error("Selected file path could not be determined.")
                return self.create_sample_resume_df()
            
            # Read the selected CSV file
            try:
                resume_df = pd.read_csv(selected_file)
                
                # Ensure required columns exist
                for col in ['File Name', 'Skills', 'Tools', 'Certifications']:
                    if col not in resume_df.columns:
                        resume_df[col] = ""
                
                return resume_df
            except Exception as e:
                st.error(f"Error reading file {selected_file}: {str(e)}")
                return self.create_sample_resume_df()
        
        except Exception as e:
            st.error(f"Error loading resume data: {str(e)}")
            return self.create_sample_resume_df()
    
    def create_sample_resume_df(self):
        """Create a sample resume DataFrame"""
        st.info("Using sample resume data")
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
        
        # Create a temporary file with a unique name
        temp_filename = f"temp_{uploaded_file.name.replace(' ', '_')}_{np.random.randint(10000)}.docx"
        temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
        
        try:
            # Write the uploaded file to a temporary file
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            
            # Extract text from the document
            doc = Document(temp_path)
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
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
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