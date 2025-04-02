import pandas as pd
import os
import re
import streamlit as st

class JobSearchUtility:
    """Utility for searching job descriptions across Excel files"""
    
    def __init__(self):
        """Initialize the job search utility"""
        self.position_report_df = None
        self.job_listings_df = None
        self.is_initialized = False
        self.pattern_detected = None
    
    def load_data_files(self, position_report_path, job_listings_path):
        """
        Load job data from Excel or CSV files
        
        Args:
            position_report_path (str): Path to the Prismforce position report file
            job_listings_path (str): Path to the Talent Recruit job listings file
            
        Returns:
            bool: True if files loaded successfully, False otherwise
        """
        try:
            # Load the Excel or CSV files based on extension
            if position_report_path.endswith('.xlsx') or position_report_path.endswith('.xls'):
                self.position_report_df = pd.read_excel(position_report_path)
            else:
                self.position_report_df = pd.read_csv(position_report_path)
                
            if job_listings_path.endswith('.xlsx') or job_listings_path.endswith('.xls'):
                self.job_listings_df = pd.read_excel(job_listings_path)
            else:
                self.job_listings_df = pd.read_csv(job_listings_path)
            
            # Log column names for debugging
            print("Position Report Columns:", self.position_report_df.columns.tolist())
            print("Job Listings Columns:", self.job_listings_df.columns.tolist())
            
            # Convert IDs to string for consistent matching
            id_columns = ['Parent Id', 'Reference Id', 'Refrence Id', 'Job Id', 'ATS Position ID', 'Project ID', 'Opportunity ID']
            
            for col in id_columns:
                if col in self.position_report_df.columns:
                    self.position_report_df[col] = self.position_report_df[col].astype(str)
                if col in self.job_listings_df.columns:
                    self.job_listings_df[col] = self.job_listings_df[col].astype(str)
            
            # Prepare data for matching
            self._preprocess_id_fields()
            
            # Identify pattern between reference IDs
            self._identify_id_patterns()
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            st.error(f"Error loading data files: {str(e)}")
            return False
    
    def _preprocess_id_fields(self):
        """
        Preprocess ID fields to handle different naming conventions
        - Normalize Parent ID in Prismforce and Reference ID in Talent Recruit
        - Handle project/opportunity ID prefixes
        """
        # Fix column naming issues (sometimes 'Refrence Id' is used instead of 'Reference Id')
        if 'Refrence Id' in self.job_listings_df.columns and 'Reference Id' not in self.job_listings_df.columns:
            self.job_listings_df['Reference Id'] = self.job_listings_df['Refrence Id']
        
        # Create normalized ID fields for matching
        if 'Parent Id' in self.position_report_df.columns:
            # Extract base Parent ID by removing project/opportunity prefixes
            self.position_report_df['Normalized_Parent_Id'] = self.position_report_df['Parent Id'].apply(
                lambda x: self._extract_base_id(str(x))
            )
        
        if 'Reference Id' in self.job_listings_df.columns:
            # Normalize Reference ID 
            self.job_listings_df['Normalized_Reference_Id'] = self.job_listings_df['Reference Id'].apply(
                lambda x: self._extract_base_id(str(x))
            )
    
    def _extract_base_id(self, id_string):
        """
        Extract the base ID from a string that might contain prefixes
        For example, "PRJ-12345" or "OPP-12345" would return "12345"
        
        Args:
            id_string (str): ID string possibly with prefix
            
        Returns:
            str: Base ID without prefix
        """
        # Define common prefixes to look for
        prefixes = ['PRJ[-_]?', 'OPP[-_]?', 'PROJ[-_]?', 'REQ[-_]?', 'JOB[-_]?']
        
        # Try to find and remove the prefix
        for prefix in prefixes:
            match = re.search(f'^{prefix}(\\d+)$', id_string)
            if match:
                return match.group(1)
        
        # If no prefix found, check if there's a dash or underscore separator
        if '-' in id_string:
            return id_string.split('-')[-1]
        if '_' in id_string:
            return id_string.split('_')[-1]
        
        # If no specific format identified, return the original
        return id_string
    
    def _identify_id_patterns(self):
        """
        Identify patterns between Prismforce Parent IDs and Talent Recruit Reference IDs
        This function identifies how the IDs relate to each other across systems
        """
        patterns = {
            'exact_match': False,
            'normalized_match': False,
            'substring_match': False,
            'ats_position_id_match': False,
            'job_id_match': False
        }
        
        # Only proceed if we have both dataframes with data
        if self.position_report_df is None or self.job_listings_df is None:
            self.pattern_detected = patterns
            return
        
        if len(self.position_report_df) == 0 or len(self.job_listings_df) == 0:
            self.pattern_detected = patterns
            return
        
        # 1. Check for direct matches between Parent Id and Reference Id
        if 'Parent Id' in self.position_report_df.columns and 'Reference Id' in self.job_listings_df.columns:
            parent_ids = set(self.position_report_df['Parent Id'].dropna().tolist())
            reference_ids = set(self.job_listings_df['Reference Id'].dropna().tolist())
            
            # Look for exact matches
            if len(parent_ids.intersection(reference_ids)) > 0:
                patterns['exact_match'] = True
        
        # 2. Check for normalized matches
        if 'Normalized_Parent_Id' in self.position_report_df.columns and 'Normalized_Reference_Id' in self.job_listings_df.columns:
            normalized_parent_ids = set(self.position_report_df['Normalized_Parent_Id'].dropna().tolist())
            normalized_reference_ids = set(self.job_listings_df['Normalized_Reference_Id'].dropna().tolist())
            
            # Look for matches in normalized IDs
            if len(normalized_parent_ids.intersection(normalized_reference_ids)) > 0:
                patterns['normalized_match'] = True
        
        # 3. Check for ATS Position ID matches
        if 'ATS Position ID' in self.position_report_df.columns and 'ATS Position ID' in self.job_listings_df.columns:
            position_ats_ids = set(self.position_report_df['ATS Position ID'].dropna().tolist())
            listings_ats_ids = set(self.job_listings_df['ATS Position ID'].dropna().tolist())
            
            if len(position_ats_ids.intersection(listings_ats_ids)) > 0:
                patterns['ats_position_id_match'] = True
        
        # 4. Check for substring matches (one ID contains the other)
        if 'Parent Id' in self.position_report_df.columns and 'Reference Id' in self.job_listings_df.columns:
            for parent_id in self.position_report_df['Parent Id'].dropna().head(50):  # Check first 50 for performance
                for ref_id in self.job_listings_df['Reference Id'].dropna().head(50):
                    if str(parent_id) in str(ref_id) or str(ref_id) in str(parent_id):
                        patterns['substring_match'] = True
                        break
                if patterns['substring_match']:
                    break
        
        # 5. Check for Job Id matches
        if 'Parent Id' in self.position_report_df.columns and 'Job Id' in self.job_listings_df.columns:
            parent_ids = set(self.position_report_df['Parent Id'].dropna().tolist())
            job_ids = set(self.job_listings_df['Job Id'].dropna().tolist())
            
            # Check for common values
            if len(parent_ids.intersection(job_ids)) > 0:
                patterns['job_id_match'] = True
        
        # Store the detected patterns
        self.pattern_detected = patterns
        
        # Log detected patterns for debugging
        print("Detected ID patterns:", patterns)
    
    def get_dropdown_options(self):
        """
        Get formatted options for the dropdown in the format: RRID[Job Id]_[Job Name]_[Client]
        
        Returns:
            list: List of formatted dropdown options
        """
        if not self.is_initialized:
            return []
        
        options = []
        
        for _, row in self.job_listings_df.iterrows():
            # Get the required fields, handling missing columns gracefully
            job_id = str(row.get('Job Id', '')) if 'Job Id' in row else ''
            ref_id = str(row.get('Reference Id', '')) if 'Reference Id' in row else ''
            if not ref_id and 'Refrence Id' in row:  # Handle misspelled column name
                ref_id = str(row.get('Refrence Id', ''))
            
            job_name = str(row.get('Job Name', '')) if 'Job Name' in row else ''
            if not job_name and 'Position' in row:  # Alternative column name
                job_name = str(row.get('Position', ''))
            if not job_name and 'Title' in row:  # Another alternative
                job_name = str(row.get('Title', ''))
                
            client = str(row.get('Client', '')) if 'Client' in row else ''
            if not client and 'Company' in row:  # Alternative column name
                client = str(row.get('Company', ''))
            
            # Handle rows with missing critical data
            if not job_id and not ref_id:
                continue
            
            # Format the dropdown option
            id_part = job_id if job_id else ref_id
            option = f"RRID{id_part}_{job_name}_{client}"
            option = option.replace('nan', '').replace('None', '')
            
            # Add to options list
            options.append(option)
        
        return options
    
    def extract_ids_from_option(self, selected_option):
        """
        Extract Job Id and other IDs from the selected dropdown option
        
        Args:
            selected_option (str): The selected dropdown option in format RRID[Job Id]_[Job Name]_[Client]
            
        Returns:
            dict: Dictionary containing extracted IDs
        """
        # Default empty values
        extracted = {
            'job_id': '',
            'job_name': '',
            'client': ''
        }
        
        # Extract Job ID from the format: RRID[Job Id]_[Job Name]_[Client]
        job_id_match = re.search(r"RRID([^_]+)_", selected_option)
        if job_id_match:
            extracted['job_id'] = job_id_match.group(1)
        
        # Extract job name and client
        parts = selected_option.split('_')
        if len(parts) >= 2:
            # Job name might be the second part (after removing RRID prefix)
            job_name_part = parts[1]
            extracted['job_name'] = job_name_part
        
        if len(parts) >= 3:
            # Client is the last part
            extracted['client'] = parts[2]
        
        return extracted
    
    def find_job_description(self, selected_option):
        """
        Find the job description for the selected option using multiple matching strategies
        
        Args:
            selected_option (str): The selected dropdown option
            
        Returns:
            tuple: (job_description, job_details_dict)
        """
        if not self.is_initialized:
            return None, None
        
        # Extract IDs from the selected option
        extracted_ids = self.extract_ids_from_option(selected_option)
        job_id = extracted_ids.get('job_id', '')
        
        if not job_id:
            return None, None
        
        # Find the matching job listing
        matching_job = self.job_listings_df[self.job_listings_df['Job Id'] == job_id]
        
        if matching_job.empty:
            # Try alternative columns if available
            if 'ID' in self.job_listings_df.columns:
                matching_job = self.job_listings_df[self.job_listings_df['ID'] == job_id]
            if matching_job.empty and 'Reference Id' in self.job_listings_df.columns:
                matching_job = self.job_listings_df[self.job_listings_df['Reference Id'] == job_id]
        
        if matching_job.empty:
            return None, None
        
        # Get reference ID from the matching job (might be spelled as 'Refrence Id')
        reference_id = None
        if 'Reference Id' in matching_job.columns:
            reference_id = matching_job['Reference Id'].iloc[0]
        elif 'Refrence Id' in matching_job.columns:
            reference_id = matching_job['Refrence Id'].iloc[0]
        
        # Get normalized reference ID if available
        normalized_reference_id = None
        if 'Normalized_Reference_Id' in matching_job.columns:
            normalized_reference_id = matching_job['Normalized_Reference_Id'].iloc[0]
        
        # Get ATS Position ID if available
        ats_position_id = None
        if 'ATS Position ID' in matching_job.columns:
            ats_position_id = matching_job['ATS Position ID'].iloc[0]
        
        # Try multiple matching strategies to find the corresponding position
        parent_match = self._find_matching_position(
            job_id=job_id, 
            reference_id=reference_id, 
            normalized_reference_id=normalized_reference_id,
            ats_position_id=ats_position_id
        )
        
        if parent_match is None or parent_match.empty:
            # As a fallback, try to create a simulated job description
            return self._create_simulated_job_description(extracted_ids), {
                'Job Id': job_id,
                'Reference Id': reference_id,
                'Job Name': extracted_ids.get('job_name', ''),
                'Client': extracted_ids.get('client', ''),
                'Status': 'Simulated - No match found in position data'
            }
        
        # Get the job description
        job_description = None
        if 'Job Description' in parent_match.columns:
            job_description = parent_match['Job Description'].iloc[0]
        elif 'Description' in parent_match.columns:
            job_description = parent_match['Description'].iloc[0]
        
        if not job_description or pd.isna(job_description):
            job_description = self._create_simulated_job_description(extracted_ids)
        
        # Create a dictionary with all relevant job details
        job_details = {
            'Job Id': job_id,
            'Reference Id': reference_id,
            'Job Name': extracted_ids.get('job_name', ''),
            'Client': extracted_ids.get('client', ''),
            'Parent Id': parent_match['Parent Id'].iloc[0] if 'Parent Id' in parent_match.columns else 'N/A',
            'ATS Position ID': ats_position_id or 'N/A'
        }
        
        return job_description, job_details
    
    def _find_matching_position(self, job_id, reference_id, normalized_reference_id, ats_position_id):
        """
        Find the matching position using multiple strategies
        
        Args:
            job_id (str): Job ID
            reference_id (str): Reference ID
            normalized_reference_id (str): Normalized Reference ID
            ats_position_id (str): ATS Position ID
            
        Returns:
            DataFrame: Matching position record or None
        """
        # Strategy 1: Match based on patterns detected
        if self.pattern_detected:
            # ATS Position ID match
            if self.pattern_detected.get('ats_position_id_match') and ats_position_id and 'ATS Position ID' in self.position_report_df.columns:
                match = self.position_report_df[self.position_report_df['ATS Position ID'] == ats_position_id]
                if not match.empty:
                    return match
            
            # Exact match between Parent Id and Reference Id
            if self.pattern_detected.get('exact_match') and reference_id and 'Parent Id' in self.position_report_df.columns:
                match = self.position_report_df[self.position_report_df['Parent Id'] == reference_id]
                if not match.empty:
                    return match
            
            # Normalized match
            if self.pattern_detected.get('normalized_match') and normalized_reference_id and 'Normalized_Parent_Id' in self.position_report_df.columns:
                match = self.position_report_df[self.position_report_df['Normalized_Parent_Id'] == normalized_reference_id]
                if not match.empty:
                    return match
            
            # Job ID match
            if self.pattern_detected.get('job_id_match') and job_id and 'Parent Id' in self.position_report_df.columns:
                match = self.position_report_df[self.position_report_df['Parent Id'] == job_id]
                if not match.empty:
                    return match
            
            # Substring match
            if self.pattern_detected.get('substring_match') and reference_id and 'Parent Id' in self.position_report_df.columns:
                matches = []
                for _, row in self.position_report_df.iterrows():
                    parent_id = str(row.get('Parent Id', ''))
                    if parent_id in reference_id or reference_id in parent_id:
                        matches.append(row)
                
                if matches:
                    return pd.DataFrame(matches)
        
        # Strategy 2: General fallback searches
        
        # Direct match with reference ID
        if reference_id and 'Parent Id' in self.position_report_df.columns:
            match = self.position_report_df[self.position_report_df['Parent Id'] == reference_id]
            if not match.empty:
                return match
        
        # Partial match with reference ID
        if reference_id and 'Parent Id' in self.position_report_df.columns:
            matches = []
            for _, row in self.position_report_df.iterrows():
                parent_id = str(row.get('Parent Id', ''))
                if reference_id in parent_id:
                    matches.append(row)
            
            if matches:
                return pd.DataFrame(matches)
        
        # Last resort: try to match based on normalized values
        if normalized_reference_id:
            matches = []
            for _, row in self.position_report_df.iterrows():
                parent_id = str(row.get('Parent Id', ''))
                normalized_parent = self._extract_base_id(parent_id)
                if normalized_reference_id == normalized_parent:
                    matches.append(row)
            
            if matches:
                return pd.DataFrame(matches)
        
        # No match found
        return None
    
    def _create_simulated_job_description(self, extracted_ids):
        """
        Create a simulated job description when no match is found
        
        Args:
            extracted_ids (dict): Dictionary of extracted IDs
            
        Returns:
            str: Simulated job description
        """
        job_name = extracted_ids.get('job_name', 'Unnamed Position')
        client = extracted_ids.get('client', 'Unknown Client')
        
        # Create a generic job description based on the job name
        keywords = re.findall(r'\w+', job_name.lower())
        
        # Define skill sets based on common job types
        skill_sets = {
            'developer': [
                'Programming skills in relevant languages',
                'Software development methodologies',
                'Problem-solving and debugging skills',
                'Code versioning tools (e.g., Git)',
                'Experience with APIs and web services'
            ],
            'engineer': [
                'Engineering principles and practices',
                'Technical documentation and specifications',
                'Problem-solving and analytical skills',
                'Project management',
                'Attention to detail'
            ],
            'data': [
                'Data analysis and interpretation',
                'Database management and SQL',
                'Statistical analysis tools',
                'Data visualization',
                'Big data technologies'
            ],
            'manager': [
                'Team leadership and management',
                'Project planning and execution',
                'Communication and presentation skills',
                'Budget management',
                'Strategic thinking'
            ],
            'designer': [
                'Design principles and techniques',
                'Creative thinking',
                'User experience (UX) design',
                'Design software proficiency',
                'Visual communication skills'
            ]
        }
        
        # Determine which skill set to use
        selected_skills = skill_sets.get('engineer')  # Default
        for keyword, skills in skill_sets.items():
            if any(keyword in kw for kw in keywords):
                selected_skills = skills
                break
        
        # Generate the job description
        job_description = f"""
        {job_name}
        
        Company: {client}
        
        Job Description:
        We are seeking a qualified {job_name} to join our team. The successful candidate will be responsible for performing various duties related to {' '.join(keywords)} and will contribute to the overall success of our projects.
        
        Responsibilities:
        - Design, develop, and implement solutions based on requirements
        - Collaborate with cross-functional teams
        - Troubleshoot and resolve technical issues
        - Document processes and technical specifications
        - Stay updated on industry trends and technologies
        
        Requirements:
        - {selected_skills[0]}
        - {selected_skills[1]}
        - {selected_skills[2]}
        - {selected_skills[3]}
        - {selected_skills[4]}
        
        Education and Experience:
        - Bachelor's degree in a relevant field
        - 3+ years of experience in a similar role
        - Proven track record of successful project delivery
        
        Note: This is a simulated job description generated because the actual description was not available.
        """
        
        return job_description


def find_data_files():
    """Find CSV and Excel files in the working directory that might contain job data"""
    data_files = [f for f in os.listdir() if f.endswith(('.csv', '.xlsx', '.xls'))]
    
    position_report_candidates = [f for f in data_files if 'position' in f.lower() or 'report' in f.lower()]
    job_listing_candidates = [f for f in data_files if 'job' in f.lower() or 'listing' in f.lower()]
    
    # If no specific matches, return all files
    if not position_report_candidates:
        position_report_candidates = data_files
    
    if not job_listing_candidates:
        job_listing_candidates = data_files
    
    return position_report_candidates, job_listing_candidates


def render_job_search_section(state_manager):
    """
    Render the job search section in the UI
    
    Args:
        state_manager: State manager instance
    
    Returns:
        bool: Whether job search is initialized
    """
    from ui.common import display_subsection_header, display_warning_message, display_success_message
    
    display_subsection_header("Job Description Search")
    
    # Get job search utility from state manager
    job_search = state_manager.get('job_search_utility')
    job_search_initialized = state_manager.get('job_search_initialized', False)
    
    # Ensure job search is properly initialized
    if not job_search:
        job_search = JobSearchUtility()
        state_manager.set('job_search_utility', job_search)
    
    # File selection section
    if not job_search.is_initialized and not job_search_initialized:
        # Find data files
        position_report_candidates, job_listing_candidates = find_data_files()
        
        if len(position_report_candidates) == 0 or len(job_listing_candidates) == 0:
            display_warning_message("No data files found. Please upload the position report and job listing files.")
            
            # Add file uploaders
            position_report_file = st.file_uploader(
                "Upload Position Report File",
                type=['csv', 'xlsx', 'xls'],
                key="position_report_upload"
            )
            
            job_listings_file = st.file_uploader(
                "Upload Job Listings File",
                type=['csv', 'xlsx', 'xls'],
                key="job_listings_upload"
            )
            
            if position_report_file and job_listings_file:
                # Save uploaded files with appropriate extensions
                position_file_ext = os.path.splitext(position_report_file.name)[1]
                job_file_ext = os.path.splitext(job_listings_file.name)[1]
                
                position_temp_path = f"position_report{position_file_ext}"
                job_temp_path = f"job_listings{job_file_ext}"
                
                with open(position_temp_path, "wb") as f:
                    f.write(position_report_file.getvalue())
                
                with open(job_temp_path, "wb") as f:
                    f.write(job_listings_file.getvalue())
                
                if st.button("Initialize Job Search", key="upload_init_btn"):
                    success = job_search.load_data_files(position_temp_path, job_temp_path)
                    if success:
                        state_manager.set('job_search_initialized', True)
                        display_success_message("Files loaded successfully!")
                        st.rerun()
            
            return False
        
        # Create two columns for file selection
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Select Position Report File**")
            position_file = st.selectbox(
                "File containing Parent Id and Job Description",
                options=position_report_candidates,
                key="position_report_file"
            )
        
        with col2:
            st.markdown("**Select Job Listings File**")
            job_file = st.selectbox(
                "File containing Job Id, Reference Id, etc.",
                options=job_listing_candidates,
                key="job_listings_file"
            )
        
        # Initialize button
        if st.button("Initialize Job Search", key="init_job_search"):
            with st.spinner("Loading job data..."):
                success = job_search.load_data_files(position_file, job_file)
                
                if success:
                    state_manager.set('job_search_initialized', True)
                    display_success_message("Job data loaded successfully!")
                    st.rerun()
                    return True
        
        return False
    
    # If we have already initialized job search in another tab
    elif not job_search.is_initialized and job_search_initialized:
        # Try to re-initialize with dummy data for demo purposes
        job_search.position_report_df = pd.DataFrame({
            'Parent Id': ['1001', '1002', '1003', '1004', '1005'],
            'Job Description': [
                'Software Engineer with 5+ years experience in Python and Java...',
                'Data Scientist with strong background in machine learning...',
                'DevOps Engineer with expertise in AWS and CI/CD pipelines...',
                'Frontend Developer with React.js experience...',
                'Backend Developer with Node.js and MongoDB experience...'
            ]
        })
        
        job_search.job_listings_df = pd.DataFrame({
            'Job Id': ['1001', '1002', '1003', '1004', '1005'],
            'Refrence Id': ['REF1001', 'REF1002', 'REF1003', 'REF1004', 'REF1005'],
            'Job Name': [
                'Software Engineer', 
                'Data Scientist', 
                'DevOps Engineer', 
                'Frontend Developer',
                'Backend Developer'
            ],
            'Client': [
                'TechCorp Inc.', 
                'DataAnalytics Ltd.', 
                'CloudSystems Inc.', 
                'WebApp Solutions',
                'ServerTech Inc.'
            ]
        })
        
        job_search.is_initialized = True
        state_manager.set('job_search_utility', job_search)
        display_success_message("Demo job search data initialized!")
        st.rerun()
        return True
    
    # Search section (only shown after initialization)
    else:
        # Display data statistics
        st.info(f"📊 Loaded {len(job_search.job_listings_df)} job listings and {len(job_search.position_report_df)} position records")
        return True