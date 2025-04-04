import streamlit as st
import os
import datetime
import pandas as pd
from utils.job_search import JobSearchUtility

def handle_database_search_selection(state_manager, context=""):
    """
    Handle selection of a job description from the database search
    
    Args:
        state_manager: State manager instance
        context: String identifying the calling context
        
    Returns:
        tuple: (jd_content, jd_source_name, jd_unique_id, success)
    """
    # Get the job search utility
    job_search = state_manager.get('job_search_utility')
    
    # If job search isn't initialized, initialize with dummy data
    if not job_search or not hasattr(job_search, 'is_initialized') or not job_search.is_initialized:
        job_search = JobSearchUtility()
        job_search.position_report_df = pd.DataFrame({
            'Parent Id': ['REF1001', 'REF1002', 'REF1003', 'REF1004', 'REF1005'],
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
            ],
            'Job Status': [
                'Active',
                'Active',
                'Closed',
                'On Hold',
                'Active'
            ]
        })
        
        job_search.is_initialized = True
        state_manager.set('job_search_utility', job_search)
        state_manager.set('job_search_initialized', True)
        st.success("Demo job search data initialized!")
    
    # Get dropdown options
    options = job_search.get_dropdown_options()
    
    if not options:
        st.warning("No job listings found in the database.")
        return None, None, None, False
    
    # Add a search box for filtering the dropdown
    search_term = st.text_input("Search for job by ID, name, or client:", key=f"{context}_job_search_term")
    
    # Filter options based on search term
    if search_term:
        filtered_options = [opt for opt in options if search_term.lower() in opt.lower()]
    else:
        filtered_options = options
    
    # Show the dropdown with filtered options
    if filtered_options:
        selected_option = st.selectbox(
            "Select Job:",
            options=filtered_options,
            key=f"{context}_job_search_dropdown"
        )
        
        # Find the job description
        if selected_option:
            job_description, job_details = job_search.find_job_description(selected_option)
            
            if job_description:
                # Display job details in an expander
                with st.expander("Job Details", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Job ID:** {job_details.get('Job Id', 'N/A')}")
                        st.markdown(f"**Reference ID:** {job_details.get('Reference Id', 'N/A')}")
                    with col2:
                        st.markdown(f"**Parent ID:** {job_details.get('Parent Id', 'N/A')}")
                        st.markdown(f"**ATS Position ID:** {job_details.get('ATS Position ID', 'N/A')}")
                
                # Always show the preview
                st.text_area(
                    "Job Description Preview", 
                    job_description,
                    height=250,
                    disabled=True,
                    key=f"{context}_preview_search"
                )
                
                # Use button for explicit selection
                if st.button("Use This Job Description", key=f"{context}_use_job_btn"):
                    return (
                        job_description,
                        selected_option,
                        f"db_{job_details.get('Job Id', '')}",
                        True
                    )
            else:
                st.error("Could not find job description for the selected job.")
    else:
        st.warning("No jobs match your search criteria.")
    
    return None, None, None, False

def direct_jd_implementation_in_optimization(state_manager, logger, context="jd_optimization"):
    """
    Direct implementation to be inserted into JD Optimization page
    
    Args:
        state_manager: State manager instance
        logger: Logger instance
        context: String identifying the calling context
        
    Returns:
        bool: Whether a JD was successfully selected
    """
    from ui.common import display_warning_message, display_success_message
    
    # Create a selection for the source type
    source_options = ["📁 File Selection", "📤 Upload New", "🔍 Search Database"]
    
    # Display selector for choosing JD source
    selected_source = st.radio(
        "Choose job description source:",
        options=source_options,
        horizontal=True,
        key=f"{context}_jd_source_selector"
    )
    
    # Variables to track job description source
    jd_content = None
    jd_source_name = None
    jd_unique_id = None
    success = False
    
    # Handle each source option
    if selected_source == "🔍 Search Database":
        jd_content, jd_source_name, jd_unique_id, success = handle_database_search_selection(
            state_manager, context
        )
    elif selected_source == "📁 File Selection":
        # File selection logic
        jd_directory = os.path.join(os.getcwd(), "Data/JDs")
        try:
            if not os.path.exists(jd_directory):
                os.makedirs(jd_directory, exist_ok=True)
                st.info(f"Created directory: {jd_directory}")
                return False
                
            files = [f for f in os.listdir(jd_directory) if f.endswith(('.txt', '.docx'))]
            
            if files:
                selected_file = st.selectbox(
                    "Select Job Description File", 
                    files, 
                    key=f"{context}_file_selector"
                )
                
                if selected_file:
                    # Load selected file
                    file_path = os.path.join(jd_directory, selected_file)
                    
                    try:
                        from utils.file_utils import read_job_description
                        file_content = read_job_description(file_path)
                        
                        # Preview the content
                        st.text_area(
                            "File Content Preview",
                            file_content,
                            height=200,
                            disabled=True,
                            key=f"{context}_file_preview"
                        )
                        
                        # Add button to confirm selection
                        if st.button("Use This File", key=f"{context}_use_file_btn"):
                            jd_content = file_content
                            jd_source_name = selected_file
                            jd_unique_id = f"file_{selected_file}"
                            success = True
                    except Exception as e:
                        st.error(f"Error reading file: {str(e)}")
            else:
                st.warning("No job description files found in JDs directory.")

        except Exception as e:
            st.error(f"Error accessing JDs directory: {str(e)}")
    
    elif selected_source == "📤 Upload New":
        uploaded_file = st.file_uploader(
            "Upload Job Description File", 
            type=['txt', 'docx'],
            key=f"{context}_file_uploader"
        )
        
        if uploaded_file:
            # Process uploaded file
            try:
                if uploaded_file.name.endswith('.txt'):
                    file_content = uploaded_file.getvalue().decode('utf-8')
                else:  # .docx
                    from utils.file_utils import process_uploaded_docx
                    file_content = process_uploaded_docx(uploaded_file)
                
                # Preview the content
                st.text_area(
                    "Uploaded Content Preview",
                    file_content,
                    height=200,
                    disabled=True,
                    key=f"{context}_upload_preview"
                )
                
                # Add button to confirm upload
                if st.button("Use This File", key=f"{context}_use_upload_btn"):
                    jd_content = file_content
                    jd_source_name = uploaded_file.name
                    jd_unique_id = f"upload_{uploaded_file.name}"
                    success = True
            except Exception as e:
                st.error(f"Error processing uploaded file: {str(e)}")
    
    # Update state with the selected JD if successful
    if success and jd_content and jd_source_name and jd_unique_id:
        # Update the JD repository
        state_manager.update_jd_repository('original', jd_content, source_tab=context)
        state_manager.update_jd_repository('source_name', jd_source_name, source_tab=context)
        state_manager.update_jd_repository('unique_id', jd_unique_id, source_tab=context)
        
        # Reset versions when changing JD source
        state_manager.update_jd_repository('enhanced_versions', [], source_tab=context)
        state_manager.update_jd_repository('selected_version_idx', 0, source_tab=context)
        state_manager.update_jd_repository('final_version', None, source_tab=context)
        
        # Add notification for JD selection
        state_manager.add_notification({
            'type': 'jd_selected',
            'source_name': jd_source_name,
            'origin': context,
            'timestamp': datetime.datetime.now().isoformat()
        })
        
        # Log file selection
        if logger:
            logger.log_file_selection(jd_source_name, jd_content)
        
        # Display success message
        display_success_message(f"Selected job description: {jd_source_name}")
        
        # Show JD preview
        with st.expander("View Selected Job Description", expanded=True):
            st.text_area(
                "Content", 
                jd_content, 
                height=250, 
                disabled=True,
                key=f"{context}_jd_preview"
            )
        
        return True
    
    return success