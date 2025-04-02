import os
import pandas as pd
import numpy as np
import streamlit as st

class ResumeHandler:
    """Class to handle finding and processing resume files"""
    
    def __init__(self):
        """Initialize resume handler"""
        # Possible locations for resume files
        self.possible_locations = [
            "Extracted_Resumes",
            "ExtractedResumes",
            "Resumes",
            "extracted_resumes",
            "resumes",
            "resume_data",
            "ResumeData",
            "Resume_Data"
        ]
        
        # Possible csv file patterns
        self.csv_patterns = [
            "resumes_analysis_output",
            "resume_data",
            "candidate_data",
            "extracted_data"
        ]
    
    def find_resume_files(self):
        """
        Find resume files in various possible locations
        
        Returns:
            dict: Dictionary of found resume files by type
        """
        # Check for resume folders in current directory and up to 2 levels down
        resume_folders = self._find_resume_folders()
        
        # Check for CSV files directly in current directory
        csv_files = [f for f in os.listdir() if f.endswith('.csv') and any(pattern in f.lower() for pattern in self.csv_patterns)]
        
        # Check for CSV files in found folders
        for folder in resume_folders:
            try:
                folder_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]
                csv_files.extend(folder_files)
            except Exception as e:
                print(f"Error accessing folder {folder}: {e}")
        
        # Group files by type
        resume_files = {
            "java_developer": [],
            "data_engineer": [],
            "general": []
        }
        
        # Categorize the files by type
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            if "java" in filename.lower():
                resume_files["java_developer"].append(csv_file)
            elif "data" in filename.lower() or "engineer" in filename.lower():
                resume_files["data_engineer"].append(csv_file)
            else:
                resume_files["general"].append(csv_file)
        
        return resume_files
    
    def _find_resume_folders(self):
        """
        Look for resume folders in various locations
        
        Returns:
            list: List of paths to resume folders
        """
        resume_folders = []
        
        # Check current directory
        cwd = os.getcwd()
        for location in self.possible_locations:
            folder_path = os.path.join(cwd, location)
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                resume_folders.append(folder_path)
        
        # Check one level up
        parent_dir = os.path.dirname(cwd)
        for location in self.possible_locations:
            folder_path = os.path.join(parent_dir, location)
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                resume_folders.append(folder_path)
        
        # Check subdirectories (one level down)
        try:
            for subdir in os.listdir(cwd):
                subdir_path = os.path.join(cwd, subdir)
                if os.path.isdir(subdir_path):
                    for location in self.possible_locations:
                        folder_path = os.path.join(subdir_path, location)
                        if os.path.exists(folder_path) and os.path.isdir(folder_path):
                            resume_folders.append(folder_path)
        except Exception as e:
            print(f"Error scanning subdirectories: {e}")
        
        return resume_folders
    
    def load_resume_file(self, file_path, jd_type=None):
        """
        Load resumes from a CSV file
        
        Args:
            file_path (str): Path to the CSV file
            jd_type (str, optional): Job description type for filtering
            
        Returns:
            DataFrame: DataFrame of resumes
        """
        try:
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return self.create_sample_resume_df()
            
            resume_df = pd.read_csv(file_path)
            
            # Clean up column names (in case of inconsistency)
            rename_dict = {}
            for col in resume_df.columns:
                if "name" in col.lower() or "file" in col.lower():
                    rename_dict[col] = "File Name"
                elif "skill" in col.lower():
                    rename_dict[col] = "Skills"
                elif "tool" in col.lower() or "tech" in col.lower():
                    rename_dict[col] = "Tools"
                elif "cert" in col.lower():
                    rename_dict[col] = "Certifications"
            
            if rename_dict:
                resume_df = resume_df.rename(columns=rename_dict)
            
            # Ensure required columns exist
            required_columns = ["File Name", "Skills", "Tools", "Certifications"]
            for col in required_columns:
                if col not in resume_df.columns:
                    if col == "File Name":
                        resume_df[col] = [f"Resume_{i}" for i in range(len(resume_df))]
                    else:
                        resume_df[col] = ""
            
            # Filter by JD type if specified
            if jd_type and "jd_type" in resume_df.columns:
                filtered_df = resume_df[resume_df["jd_type"] == jd_type]
                if not filtered_df.empty:
                    return filtered_df
            
            print(f"Loaded {len(resume_df)} resumes from {file_path}")
            return resume_df
        except Exception as e:
            print(f"Error loading resume file {file_path}: {e}")
            return self.create_sample_resume_df()
    
    def find_best_resume_file(self, jd_type):
        """
        Find the best matching resume file for the given JD type
        
        Args:
            jd_type (str): Job description type
            
        Returns:
            str: Path to the best matching resume file
        """
        # Find all resume files
        resume_files = self.find_resume_files()
        
        # First check for exact type match
        if jd_type in resume_files and resume_files[jd_type]:
            return resume_files[jd_type][0]
        
        # Then check for any files
        for type_key in resume_files:
            if resume_files[type_key]:
                return resume_files[type_key][0]
        
        # If no files found, create a sample directory with a sample file
        self._create_sample_resume_directory()
        return "Extracted_Resumes/sample_resumes.csv"
    
    def _create_sample_resume_directory(self):
        """Create a sample directory with sample resume data"""
        try:
            # Create directory if it doesn't exist
            os.makedirs("Extracted_Resumes", exist_ok=True)
            
            # Create sample data
            sample_data = self.create_sample_resume_df()
            
            # Save to CSV
            sample_data.to_csv("Extracted_Resumes/sample_resumes.csv", index=False)
            
            print("Created sample resume directory and file")
        except Exception as e:
            print(f"Error creating sample directory: {e}")
    
    def create_sample_resume_df(self):
        """Create a sample resume DataFrame"""
        print("Creating sample resume data")
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

# Function to use in the candidate_ranking.py file
def get_resume_data_for_jd_type(jd_type):
    """
    Get resume data for the specified JD type
    
    Args:
        jd_type (str): Job description type
        
    Returns:
        DataFrame: DataFrame of resumes
    """
    # Initialize handler
    handler = ResumeHandler()
    
    # Find best matching file
    best_file = handler.find_best_resume_file(jd_type)
    
    # Load and return the data
    return handler.load_resume_file(best_file, jd_type)