import os
import numpy as np
import pandas as pd
import tempfile
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO, StringIO
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
                str(resume.get('Certifications', ''))
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
                st.warning("No resume CSV files found.")
                return None
            
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
                return None
            
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
                return None
        
        except Exception as e:
            st.error(f"Error loading resume data: {str(e)}")
            return None
    
    def analyze_uploaded_resume(self, uploaded_file):
        """
        Analyze a user-uploaded resume (.docx or .csv) and return the extracted information.
        
        Args:
            uploaded_file (UploadedFile): The uploaded resume file
            
        Returns:
            dict: Dictionary with extracted resume details
        """
        try:
            # Process based on file extension
            if uploaded_file.name.endswith('.docx'):
                return self._analyze_docx_resume(uploaded_file)
            elif uploaded_file.name.endswith('.csv'):
                return self._analyze_csv_resume(uploaded_file)
            else:
                st.error(f"Unsupported file format: {uploaded_file.name}. Please upload DOCX or CSV files.")
                return None
        except Exception as e:
            st.error(f"Error analyzing resume {uploaded_file.name}: {str(e)}")
            return None
    
    def _analyze_docx_resume(self, uploaded_file):
        """
        Analyze a DOCX resume file
        
        Args:
            uploaded_file (UploadedFile): The uploaded DOCX file
            
        Returns:
            dict: Dictionary with extracted resume details
        """
        try:
            # Show progress indicator
            with st.spinner(f"Processing DOCX: {uploaded_file.name}"):
                # Get file content
                docx_bytes = uploaded_file.getvalue()
                st.info(f"DOCX file size: {len(docx_bytes)} bytes")
                
                # Load the document
                doc = Document(BytesIO(docx_bytes))
                
                # Extract text from paragraphs
                paragraphs = []
                for para in doc.paragraphs:
                    if para.text.strip():  # Only include non-empty paragraphs
                        paragraphs.append(para.text)
                
                resume_text = "\n".join(paragraphs)
                st.success(f"Successfully extracted {len(paragraphs)} paragraphs")
                
                # Extract information
                skills_map = extract_skills(resume_text)
                skills_str = ", ".join([item for sublist in skills_map.values() for item in sublist])
                
                # Detect tools
                tools_keywords = [
                    'git', 'docker', 'kubernetes', 'jenkins', 'jira', 
                    'confluence', 'aws', 'azure', 'vs code', 'intellij',
                    'eclipse', 'idea', 'visual studio', 'vscode', 'maven',
                    'gradle', 'npm', 'yarn', 'webpack', 'jupyter'
                ]
                
                detected_tools = []
                for tool in tools_keywords:
                    if tool.lower() in resume_text.lower():
                        detected_tools.append(tool)
                
                # Detect certifications
                cert_keywords = [
                    'certified', 'certification', 'certificate', 'aws', 'azure', 
                    'google', 'professional', 'associate', 'expert', 'oracle',
                    'microsoft', 'java', 'python', 'scrum', 'pmp'
                ]
                
                certification_text = ""
                for line in paragraphs:
                    if any(kw.lower() in line.lower() for kw in cert_keywords):
                        certification_text += line + "\n"
                
                if not certification_text:
                    certification_text = "None specified"
                
                return {
                    'File Name': uploaded_file.name,
                    'Skills': skills_str or "General technical skills",
                    'Tools': ", ".join(detected_tools) or "Standard development tools",
                    'Certifications': certification_text
                }
        except Exception as e:
            st.error(f"Error processing DOCX file: {str(e)}")
            return None
    
    def _analyze_csv_resume(self, uploaded_file):
        """
        Process a CSV resume file
        
        Args:
            uploaded_file (UploadedFile): The uploaded CSV file
            
        Returns:
            dict: Dictionary with extracted resume details or None if CSV has multiple rows
        """
        try:
            # Show progress indicator
            with st.spinner(f"Processing CSV: {uploaded_file.name}"):
                # Read CSV content
                csv_content = uploaded_file.getvalue().decode('utf-8')
                
                # Parse CSV
                import csv
                
                # First check if this is a multi-row CSV (resume dataset) or single resume
                csv_rows = list(csv.reader(StringIO(csv_content)))
                
                if len(csv_rows) <= 2:  # Header + single data row
                    # This is likely a single resume in CSV format
                    if len(csv_rows) == 2:
                        # Has header and one data row
                        header = csv_rows[0]
                        data = csv_rows[1]
                        
                        # Create a dictionary of header->value
                        resume_data = {header[i]: data[i] for i in range(min(len(header), len(data)))}
                        
                        # Extract required fields or use defaults
                        return {
                            'File Name': uploaded_file.name,
                            'Skills': resume_data.get('Skills', resume_data.get('skills', '')),
                            'Tools': resume_data.get('Tools', resume_data.get('tools', '')),
                            'Certifications': resume_data.get('Certifications', resume_data.get('certifications', ''))
                        }
                    else:
                        # No data rows, just header
                        st.warning(f"CSV file {uploaded_file.name} contains only a header row with no data")
                        return None
                else:
                    # This is a multi-row CSV, likely a resume dataset
                    # Parse with pandas
                    df = pd.read_csv(StringIO(csv_content))
                    st.success(f"CSV file contains {len(df)} resumes. Processing as a resume dataset.")
                    
                    # Return None to indicate this should be treated as a full dataset not a single resume
                    return None
        except Exception as e:
            st.error(f"Error processing CSV file: {str(e)}")
            return None
    
    def process_resume_pool(self, uploaded_files):
        """
        Process a batch of uploaded resume files and return a DataFrame
        
        Args:
            uploaded_files (list): List of uploaded resume files
            
        Returns:
            DataFrame: DataFrame containing processed resume data
        """
        processed_resumes = []
        csv_dataframes = []
        
        # First pass: Process each file
        for uploaded_file in uploaded_files:
            try:
                if uploaded_file.name.endswith(".docx"):
                    # Process DOCX as individual resume
                    resume_data = self.analyze_uploaded_resume(uploaded_file)
                    if resume_data is not None:
                        processed_resumes.append(resume_data)
                        st.success(f"Successfully processed DOCX: {uploaded_file.name}")
                    else:
                        st.warning(f"Could not extract data from {uploaded_file.name}")
                
                elif uploaded_file.name.endswith(".csv"):
                    # Check if CSV is a single resume or a dataset
                    resume_data = self._analyze_csv_resume(uploaded_file)
                    
                    if resume_data is not None:
                        # This is a single resume in CSV format
                        processed_resumes.append(resume_data)
                        st.success(f"Successfully processed single-resume CSV: {uploaded_file.name}")
                    else:
                        # This might be a multi-resume CSV dataset
                        try:
                            df = pd.read_csv(BytesIO(uploaded_file.getvalue()))
                            
                            # Check if it has the required columns
                            required_cols = ['File Name', 'Skills', 'Tools']
                            missing_cols = [col for col in required_cols if col not in df.columns]
                            
                            if missing_cols:
                                # Try lowercase column names
                                lowercase_cols = {col.lower(): col for col in df.columns}
                                renamed_cols = {}
                                
                                for required in required_cols:
                                    if required.lower() in lowercase_cols:
                                        renamed_cols[lowercase_cols[required.lower()]] = required
                                
                                if renamed_cols:
                                    df = df.rename(columns=renamed_cols)
                                    
                                    # Check again for missing columns
                                    missing_cols = [col for col in required_cols if col not in df.columns]
                            
                            if not missing_cols:
                                # This is a valid resume dataset
                                csv_dataframes.append(df)
                                st.success(f"Successfully loaded resume dataset: {uploaded_file.name} with {len(df)} entries")
                            else:
                                st.warning(f"CSV file {uploaded_file.name} is missing required columns: {', '.join(missing_cols)}")
                        except Exception as e:
                            st.error(f"Error processing multi-resume CSV {uploaded_file.name}: {str(e)}")
                
                else:
                    st.warning(f"Skipping {uploaded_file.name} - not a supported format (.docx or .csv)")
            
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        # Second pass: Combine results
        if csv_dataframes:
            # If we have CSV datasets, use those first
            combined_df = pd.concat(csv_dataframes, ignore_index=True)
            
            # Add any individually processed resumes
            if processed_resumes:
                individual_df = pd.DataFrame(processed_resumes)
                combined_df = pd.concat([combined_df, individual_df], ignore_index=True)
            
            # Ensure all required columns exist
            for col in ['File Name', 'Skills', 'Tools', 'Certifications']:
                if col not in combined_df.columns:
                    combined_df[col] = ""
            
            return combined_df
            
        elif processed_resumes:
            # If we only have individually processed resumes
            return pd.DataFrame(processed_resumes)
        
        # No valid data
        return None