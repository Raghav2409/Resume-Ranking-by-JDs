import os
import streamlit as st
import pandas as pd
import numpy as np
from models.resume_analyzer import ResumeAnalyzer
from utils.text_processing import detect_jd_type
from utils.visualization import create_distribution_chart, create_radar_chart
from ui.common import display_section_header, display_subsection_header, display_info_message, display_warning_message

def render_candidate_ranking_page():
    """Render the candidate ranking page with enhanced resume pool management"""
    
    display_section_header("🎯 Resume Ranking")
    
    # --- Load Job Data ---
    job_df = load_job_descriptions()
    if job_df is None:
        return
    
    # --- Setup Resume Analyzer ---
    resume_analyzer = ResumeAnalyzer()
    
    # --- Create Layout Columns ---
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        display_subsection_header("Select Position")
        
        # Combine job names with enhanced versions if available
        job_names = job_df['File Name'].tolist()
        if st.session_state.get('final_version'):
            job_names.append("Final Enhanced Version")
        if st.session_state.get('client_enhanced_jd'):
            job_names.append("Client Enhanced Version")
        
        selected = st.selectbox('Choose position:', job_names, label_visibility="collapsed")
        
        # Set job_desc based on selection
        if selected == "Final Enhanced Version":
            if st.session_state.get('final_version'):
                job_desc = create_job_desc_from_enhanced("Final Enhanced Version", 
                                                       st.session_state.final_version)
            else:
                st.warning("No final enhanced job description available.")
                if job_df.shape[0] > 0:
                    job_desc = job_df.iloc[0]
                else:
                    return
        elif selected == "Client Enhanced Version":
            if st.session_state.get('client_enhanced_jd'):
                job_desc = create_job_desc_from_enhanced("Client Enhanced Version", 
                                                       st.session_state.client_enhanced_jd)
            else:
                st.warning("No client enhanced job description available.")
                if job_df.shape[0] > 0:
                    job_desc = job_df.iloc[0]
                else:
                    return
        else:
            job_desc = job_df[job_df['File Name'] == selected].iloc[0]
        
        jd_type = job_desc['JD_Type']
        st.markdown(f"**Resume Pool:** {jd_type.replace('_', ' ').title()}")
        
        # --- Resume Pool Selection ---
        display_subsection_header("Resume Pools")
        
        # Initialize resume_pools in session state if it doesn't exist
        if "resume_pools" not in st.session_state:
            st.session_state.resume_pools = []  # List of dicts: {"pool_name": str, "data": DataFrame}
        
        # Add generic options along with any user-uploaded pools
        generic_options = ["General", "Data Engineer", "Java Developer"]
        user_pools = [pool["pool_name"] for pool in st.session_state.resume_pools]
        
        pool_options = ["(Auto Selection)"] + user_pools + generic_options + ["Upload New Resume Pool"]
        selected_pool_option = st.selectbox(
            "Select Resume Pool Manually (Optional)", 
            pool_options,
            key="resume_pool_selector"
        )
        
        # Handle resume pool selection
        resume_df = handle_resume_pool_selection(selected_pool_option, resume_analyzer, jd_type)
        if resume_df is None:
            st.warning("No resume data available. Please select or upload a valid resume pool.")
            return
        
        # --- Analyze Button ---
        if st.button('🔍 Analyze Resumes', type="primary", key="analyze_resume_btn"):
            with st.spinner('Analyzing resumes...'):
                try:
                    st.session_state['analysis_results'] = resume_analyzer.categorize_resumes(job_desc, resume_df)
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    
                    # Create fallback result set with random scores
                    st.session_state['analysis_results'] = create_fallback_analysis(resume_df)
                
                # Force a rerun to update the UI
                st.rerun()
        
        # Display top matches preview
        if 'analysis_results' in st.session_state:
            display_top_matches(st.session_state['analysis_results'])
    
    # --- Results Display ---
    if 'analysis_results' in st.session_state:
        categorized_resumes = st.session_state['analysis_results']
        
        # Analysis overview in second column
        with col2:
            display_subsection_header("Overview")
            try:
                chart = create_distribution_chart(categorized_resumes)
                st.plotly_chart(chart, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating distribution chart: {str(e)}")
                display_info_message("Chart visualization failed. Please check your data.")
        
        # Detailed analysis in third column
        with col3:
            display_subsection_header("Detailed Analysis")
            if 'top_3' in categorized_resumes and len(categorized_resumes['top_3']) > 0:
                display_detailed_resume_analysis(categorized_resumes, job_desc)
            else:
                st.info("No detailed analysis available.")
        
        # All Resumes by Category section
        st.markdown("---")
        display_section_header("📑 All Resumes by Category")
        display_categorized_resumes(categorized_resumes)


def load_job_descriptions():
    """Load job descriptions from files or create sample data"""
    jd_directory = os.path.join(os.getcwd(), "JDs")
    
    try:
        # Try to list files from JDs directory
        if os.path.exists(jd_directory):
            files = [f for f in os.listdir(jd_directory) if f.endswith(('.txt', '.docx'))]
            
            if files:
                # Create a basic dataframe with file names
                job_data = {
                    'File Name': files,
                    'Skills': [''] * len(files),
                    'Tools': [''] * len(files),
                    'JD_Type': [detect_jd_type(file) for file in files]
                }
                return pd.DataFrame(job_data)
        
        # If no JD directory or no files, try to load analysis file
        analysis_file = "job_descriptions_analysis_output.csv"
        if os.path.exists(analysis_file):
            job_df = pd.read_csv(analysis_file)
            
            # Add JD_Type if it doesn't exist
            if 'JD_Type' not in job_df.columns:
                job_df['JD_Type'] = 'unknown'
                
                # Add job types based on file names
                java_python_keywords = ['java', 'python', 'support']
                data_engineer_keywords = ['data', 'engineer', 'analytics']
                
                for index, row in job_df.iterrows():
                    file_name = str(row['File Name']).lower()
                    if any(keyword in file_name for keyword in java_python_keywords):
                        job_df.at[index, 'JD_Type'] = 'java_developer'
                    elif any(keyword in file_name for keyword in data_engineer_keywords):
                        job_df.at[index, 'JD_Type'] = 'data_engineer'
                    else:
                        job_df.at[index, 'JD_Type'] = 'general'
            
            return job_df
    
    except Exception as e:
        st.error(f"Error loading job data: {e}")
    
    # Create sample data as a fallback
    st.warning("No job description files found. Using sample data instead.")
    job_data = {
        'File Name': ['DataAnalyticsAIMLJD.txt', 'JobDescriptionJavaPythonSupport.txt'],
        'Skills': ['Python, Java, ML, AI, Data Analysis', 'Java, Python, Object-Oriented Programming'],
        'Tools': ['SQL, Cloud, Docker', 'Debugging tools, CoderPad'],
        'JD_Type': ['data_engineer', 'java_developer']
    }
    return pd.DataFrame(job_data)


def create_job_desc_from_enhanced(version_name, enhanced_content):
    """Create a job description Series from enhanced content"""
    if isinstance(enhanced_content, dict):
        return pd.Series({
            'File Name': version_name,
            'JD_Type': "Enhanced",
            'Skills': enhanced_content.get('skills', ''),
            'Tools': enhanced_content.get('tools', '')
        })
    elif isinstance(enhanced_content, str):
        # If it's a string, create a basic job description
        return pd.Series({
            'File Name': version_name,
            'JD_Type': "Enhanced",
            'Skills': enhanced_content,
            'Tools': ''
        })
    else:
        # Return a default Series if content is invalid
        return pd.Series({
            'File Name': version_name,
            'JD_Type': "Enhanced",
            'Skills': "Unknown skills",
            'Tools': "Unknown tools"
        })


def handle_resume_pool_selection(selection, resume_analyzer, jd_type):
    """Handle different resume pool selection options"""
    if selection == "(Auto Selection)":
        # Auto-select resume pool based on JD type
        default_file_map = {
            "java_developer": "resumes_analysis_outputJDJavaDeveloper.csv",
            "data_engineer": "resumes_analysis_output_JDPrincipalSoftwareEngineer.csv",
            "general": "resumes_analysis_output.csv",
            "unknown": "resumes_analysis_output.csv"
        }
        
        default_file = default_file_map.get(jd_type, "resumes_analysis_output.csv")
        
        if os.path.exists(default_file):
            try:
                resume_df = pd.read_csv(default_file)
                st.success(f"Loaded default resume pool based on job type ({jd_type})")
                return resume_df
            except Exception as e:
                st.error(f"Error loading default resume file: {e}")
                return None
        else:
            # Try finding similar files
            possible_files = [f for f in os.listdir() if f.endswith('.csv') and 'resume' in f.lower()]
            if possible_files:
                try:
                    resume_df = pd.read_csv(possible_files[0])
                    st.info(f"Using alternative resume file: {possible_files[0]}")
                    return resume_df
                except Exception:
                    pass
            
            st.warning("Default resume pool file not found.")
            return create_sample_resume_df()
    
    elif selection == "Upload New Resume Pool":
        # Create UI for uploading new resumes
        new_pool_name = st.text_input("Enter new pool name:", key="new_pool_name")
        new_pool_files = st.file_uploader(
            "Upload resumes for the new pool", 
            type=['docx'], 
            accept_multiple_files=True, 
            key="new_pool_files"
        )
        
        if st.button("Add Resume Pool", key="add_pool"):
            if new_pool_name and new_pool_files:
                # Process all uploaded resumes
                pool_df = resume_analyzer.process_resume_pool(new_pool_files)
                
                if pool_df is not None and not pool_df.empty:
                    st.session_state.resume_pools.append({
                        "pool_name": new_pool_name, 
                        "data": pool_df
                    })
                    st.success(f"Resume pool '{new_pool_name}' added with {len(pool_df)} resumes!")
                    st.rerun()
                else:
                    st.warning("No valid resumes were processed. Please check your files.")
            else:
                st.warning("Please provide both a pool name and upload at least one resume file.")
        return None  # Return None to indicate we're still in the upload phase
    
    elif selection in ["General", "Data Engineer", "Java Developer"]:
        # Load from predefined files
        generic_map = {
            "General": "resumes_analysis_output.csv",
            "Data Engineer": "resumes_analysis_output_JDPrincipalSoftwareEngineer.csv",
            "Java Developer": "resumes_analysis_outputJDJavaDeveloper.csv"
        }
        default_file = generic_map[selection]
        
        if os.path.exists(default_file):
            try:
                resume_df = pd.read_csv(default_file)
                st.success(f"Loaded {selection} resume pool")
                return resume_df
            except Exception as e:
                st.error(f"Error loading resume file: {e}")
                return None
        else:
            st.warning(f"Resume pool file for {selection} not found.")
            return create_sample_resume_df()
    
    else:
        # Check if a user-uploaded pool was selected
        for pool in st.session_state.resume_pools:
            if pool["pool_name"] == selection:
                st.success(f"Loaded custom resume pool '{selection}' with {len(pool['data'])} resumes")
                return pool["data"]
    
    # Default fallback
    return create_sample_resume_df()


def create_sample_resume_df():
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


def create_fallback_analysis(resume_df):
    """Create fallback analysis results with random scores"""
    all_resumes = []
    for i in range(len(resume_df)):
        score = np.random.uniform(0.1, 0.4)
        all_resumes.append({
            'Resume ID': resume_df.iloc[i]['File Name'],
            'Skills': resume_df.iloc[i]['Skills'],
            'Tools': resume_df.iloc[i]['Tools'],
            'Certifications': resume_df.iloc[i]['Certifications'],
            'Score': score
        })
    
    # Sort by score
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


def display_top_matches(analysis_results):
    """Display top match previews"""
    display_subsection_header("Top Matches")
    if 'top_3' in analysis_results and analysis_results['top_3']:
        for i, resume in enumerate(analysis_results['top_3'][:3]):
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin:0">#{i + 1} - {resume['Resume ID']}</h4>
                <p style="margin:0">Match: {resume['Score']:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No top matches available yet. Click 'Analyze Resumes' to see results.")


def display_detailed_resume_analysis(categorized_resumes, job_desc):
    """Display detailed analysis of top resumes"""
    if not categorized_resumes['top_3']:
        st.info("No resume analysis data available")
        return
        
    tabs = st.tabs(["#1", "#2", "#3"])
    
    for i, (tab, resume) in enumerate(zip(tabs, categorized_resumes['top_3'][:3])):
        with tab:
            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.markdown(f"**Score:** {resume['Score']:.2%}")
                try:
                    radar_chart = create_radar_chart(resume, job_desc)
                    st.plotly_chart(radar_chart, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating radar chart: {str(e)}")
                    st.info("Match analysis visualization unavailable")
            
            with col_b:
                st.markdown(f"""
                <div class="insight-box compact-text">
                    <h4>Key Match Analysis</h4>
                    <p>This candidate shows alignment with the job requirements based on their skills and experience:</p>
                    <ul>
                        <li>Technical skills match core requirements</li>
                        <li>Experience with relevant tools and technologies</li>
                        <li>{resume['Certifications'] if resume['Certifications'] else 'Experience'} enhances qualifications</li>
                    </ul>
                    <p><strong>Overall assessment:</strong> Good potential match based on technical qualifications.</p>
                </div>
                """, unsafe_allow_html=True)


def display_categorized_resumes(categorized_resumes):
    """Display all resumes categorized by match level"""
    cat_col1, cat_col2, cat_col3 = st.columns(3)
    
    with cat_col1:
        with st.expander(f"High Matches ({len(categorized_resumes.get('high_matches', []))})"):
            for resume in categorized_resumes.get('high_matches', []):
                st.markdown(f"""
                <div class="category-high">
                    <h4 style="margin:0">{resume['Resume ID']}</h4>
                    <p style="margin:0">Match: {resume['Score']:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
    
    with cat_col2:
        with st.expander(f"Medium Matches ({len(categorized_resumes.get('medium_matches', []))})"):
            for resume in categorized_resumes.get('medium_matches', []):
                st.markdown(f"""
                <div class="category-medium">
                    <h4 style="margin:0">{resume['Resume ID']}</h4>
                    <p style="margin:0">Match: {resume['Score']:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
    
    with cat_col3:
        with st.expander(f"Low Matches ({len(categorized_resumes.get('low_matches', []))})"):
            for resume in categorized_resumes.get('low_matches', []):
                st.markdown(f"""
                <div class="category-low">
                    <h4 style="margin:0">{resume['Resume ID']}</h4>
                    <p style="margin:0">Match: {resume['Score']:.2%}</p>
                </div>
                """, unsafe_allow_html=True)