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
            position_report_path (str): Path to the position report file
            job_listings_path (str): Path to the job listings file
            
        Returns:
            bool: True if files loaded successfully, False otherwise
        """
        try:
            # Load the Excel or CSV files based on extension
            if position_report_path.endswith('Data/data Set/ Position Report/.xlsx') or position_report_path.endswith('Data/data Set/ Position Report/.xls'):
                self.position_report_df = pd.read_excel(position_report_path)
            else:
                self.position_report_df = pd.read_csv(position_report_path)
                
            if job_listings_path.endswith('Data/Data Set/ Job Listing/.xlsx') or job_listings_path.endswith('Data/Data Set/ Job Listing/.xls'):
                self.job_listings_df = pd.read_excel(job_listings_path)
            else:
                self.job_listings_df = pd.read_csv(job_listings_path)
            
            # Convert IDs to string for consistent matching
            if 'Parent Id' in self.position_report_df.columns:
                self.position_report_df['Parent Id'] = self.position_report_df['Parent Id'].astype(str)
            
            if 'Job Id' in self.job_listings_df.columns:
                self.job_listings_df['Job Id'] = self.job_listings_df['Job Id'].astype(str)
                
            if 'Refrence Id' in self.job_listings_df.columns:
                self.job_listings_df['Refrence Id'] = self.job_listings_df['Refrence Id'].astype(str)
            
            # Identify pattern between Job ID/Reference ID and Parent ID
            self._identify_id_pattern()
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            st.error(f"Error loading data files: {str(e)}")
            return False
    
    def _identify_id_pattern(self):
        """
        Identify patterns between Job ID/Reference ID and Parent ID
        This function analyzes how the IDs relate to each other
        """
        patterns = {
            'ats_position_id_match': False,
            'direct_match': False,
            'reference_id_match': False,
            'parent_id_format': None,
            'job_id_in_parent': False,
            'reference_id_in_parent': False
        }
        
        # Check for ATS Position ID in both datasets
        if (len(self.position_report_df) > 0 and len(self.job_listings_df) > 0):
            # Check if ATS Position ID exists in both datasets
            if 'ATS Position ID' in self.position_report_df.columns and 'ATS Position ID' in self.job_listings_df.columns:
                # Convert to string for consistent matching
                self.position_report_df['ATS Position ID'] = self.position_report_df['ATS Position ID'].astype(str)
                self.job_listings_df['ATS Position ID'] = self.job_listings_df['ATS Position ID'].astype(str)
                
                # Flag that we can use ATS Position ID for matching
                patterns['ats_position_id_match'] = True
                
        # Fallback patterns if ATS Position ID isn't available
        if not patterns['ats_position_id_match'] and 'Parent Id' in self.position_report_df.columns and 'Job Id' in self.job_listings_df.columns:
            # Sample some IDs for comparison
            parent_ids = self.position_report_df['Parent Id'].sample(min(10, len(self.position_report_df))).tolist()
            job_ids = self.job_listings_df['Job Id'].sample(min(10, len(self.job_listings_df))).tolist()
            
            # Check if Job ID directly matches Parent ID
            for job_id in job_ids:
                if job_id in parent_ids:
                    patterns['direct_match'] = True
                    break
            
            # Check if Reference ID exists and matches Parent ID
            if 'Refrence Id' in self.job_listings_df.columns:
                reference_ids = self.job_listings_df['Refrence Id'].sample(min(10, len(self.job_listings_df))).tolist()
                for ref_id in reference_ids:
                    if ref_id in parent_ids:
                        patterns['reference_id_match'] = True
                        break
            
            # Check if Parent IDs contain Job IDs
            for parent_id in parent_ids:
                for job_id in job_ids:
                    if job_id in parent_id:
                        patterns['job_id_in_parent'] = True
                        break
            
            # Check if Parent IDs contain Reference IDs
            if 'Refrence Id' in self.job_listings_df.columns:
                for parent_id in parent_ids:
                    for ref_id in reference_ids:
                        if ref_id in parent_id:
                            patterns['reference_id_in_parent'] = True
                            break
            
            # Look for common patterns in Parent IDs
            parent_id_patterns = []
            for parent_id in parent_ids:
                # Check for formats like X-Y-Z, X/Y/Z, etc.
                if '-' in parent_id:
                    parent_id_patterns.append('dash_separated')
                elif '/' in parent_id:
                    parent_id_patterns.append('slash_separated')
            
            if parent_id_patterns:
                most_common = max(set(parent_id_patterns), key=parent_id_patterns.count)
                patterns['parent_id_format'] = most_common
        
        self.pattern_detected = patterns
    
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
            ref_id = str(row.get('Refrence Id', '')) if 'Refrence Id' in row else ''
            job_name = str(row.get('Job Name', '')) if 'Job Name' in row else ''
            client = str(row.get('Client', '')) if 'Client' in row else ''
            
            # Format the dropdown option
            option = f"RRID{job_id}_{job_name}_{client}"
            
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
        Find the job description for the selected option
        
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
            return None, None
        
        # Get reference ID from the matching job
        reference_id = matching_job['Refrence Id'].iloc[0] if 'Refrence Id' in matching_job.columns else None
        
        # Get ATS Position ID if available
        ats_position_id = None
        if 'ATS Position ID' in matching_job.columns:
            ats_position_id = matching_job['ATS Position ID'].iloc[0]
        
        # Primary Strategy: Use ATS Position ID if available in both datasets
        parent_match = None
        if ats_position_id and 'ATS Position ID' in self.position_report_df.columns:
            parent_match = self.position_report_df[self.position_report_df['ATS Position ID'] == ats_position_id]
        
        # Fallback strategies if ATS Position ID doesn't yield a match
        if parent_match is None or parent_match.empty:
            # Strategy 1: Try direct match between Job Id and Parent Id
            parent_match = self.position_report_df[self.position_report_df['Parent Id'] == job_id]
            
            # Strategy 2: Try match between Reference Id and Parent Id
            if parent_match.empty and reference_id:
                parent_match = self.position_report_df[self.position_report_df['Parent Id'] == reference_id]
            
            # Strategy 3: Try partial match where Parent Id contains Job Id
            if parent_match.empty:
                parent_match = self.position_report_df[self.position_report_df['Parent Id'].str.contains(job_id, na=False)]
            
            # Strategy 4: Try partial match where Parent Id contains Reference Id
            if parent_match.empty and reference_id:
                parent_match = self.position_report_df[self.position_report_df['Parent Id'].str.contains(reference_id, na=False)]
        
        if parent_match.empty:
            return None, None
        
        # Get the job description
        job_description = parent_match['Job Description'].iloc[0] if 'Job Description' in parent_match.columns else None
        
        # Use dummy job description if not found (for demo purposes)
        if not job_description:
            job_description = f"""
            Job Description for {extracted_ids.get('job_name', 'Position')}
            
            Company: {extracted_ids.get('client', 'Our Client')}
            
            Responsibilities:
            - Develop high-quality software design and architecture
            - Identify, prioritize and execute tasks in the software development lifecycle
            - Develop tools and applications by producing clean, efficient code
            - Automate tasks through appropriate tools and scripting
            - Review and debug code
            - Perform validation and verification testing
            - Collaborate with internal teams and vendors to fix and improve products
            - Document development phases and monitor systems
            
            Requirements:
            - Proven experience as a Software Developer
            - Experience with development tools and languages
            - Problem-solving abilities and critical thinking
            - Excellent communication skills
            - Teamwork skills with a quality-oriented mindset
            - BS/MS degree in Computer Science, Engineering or a related field
            """
        
        # Create a dictionary with all relevant job details
        job_details = {
            'Job Id': job_id,
            'Reference Id': reference_id,
            'Job Name': extracted_ids.get('job_name', ''),
            'Client': extracted_ids.get('client', ''),
            'Parent Id': parent_match['Parent Id'].iloc[0] if 'Parent Id' in parent_match.columns else 'N/A',
            'ATS Position ID': ats_position_id or 'N/A',
            'Job Description': job_description
        }
        
        return job_description, job_details


def find_data_files():
    """Find CSV and Excel files in the specified directories that might contain job data"""
    directories = ['Data/Data Set/Job Listing', 'Data/Data Set/Position Report']
    
    data_files = []
    position_report_candidates = []
    job_listing_candidates = []
    
    for directory in directories:
        for f in os.listdir(directory):
            if f.endswith(('.csv', '.xlsx', '.xls')):
                file_path = os.path.join(directory, f)
                data_files.append(file_path)
                
                if 'position' in f.lower() or 'report' in f.lower():
                    position_report_candidates.append(file_path)
                if 'job' in f.lower() or 'listing' in f.lower():
                    job_listing_candidates.append(file_path)
    
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