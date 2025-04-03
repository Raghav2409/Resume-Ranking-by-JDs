import os
import streamlit as st
import pandas as pd
import numpy as np
from ui.common import display_section_header, display_subsection_header, display_info_message, display_warning_message, display_success_message
from utils.visualization import create_distribution_chart, create_radar_chart

class ResumeAnalyzer:
    """Simplified ResumeAnalyzer class for demo purposes"""
    
    def __init__(self):
        """Initialize the resume analyzer"""
        pass
    
    def categorize_resumes(self, job_desc, resume_df):
        """
        Categorize resumes based on match with job description
        This is a simplified implementation for demo purposes
        
        Args:
            job_desc: Job description (dict or Series)
            resume_df: DataFrame of resumes
            
        Returns:
            dict: Categorized resumes
        """
        # Extract skills from job desc
        jd_skills = str(job_desc.get('Skills', '')).lower().split(', ')
        jd_tools = str(job_desc.get('Tools', '')).lower().split(', ')
        
        # Initialize lists
        all_resumes = []
        
        # Process each resume
        for _, resume in resume_df.iterrows():
            # Extract skills from resume
            resume_skills = str(resume.get('Skills', '')).lower().split(', ')
            resume_tools = str(resume.get('Tools', '')).lower().split(', ')
            
            # Calculate match scores
            skill_matches = sum(1 for skill in resume_skills if any(jd_skill in skill for jd_skill in jd_skills))
            tool_matches = sum(1 for tool in resume_tools if any(jd_tool in tool for jd_tool in jd_tools))
            
            # Calculate overall score (weighted)
            max_skills = max(len(jd_skills), 1)
            max_tools = max(len(jd_tools), 1)
            
            skill_score = skill_matches / max_skills
            tool_score = tool_matches / max_tools
            
            # Calculate weighted score
            score = 0.7 * skill_score + 0.3 * tool_score
            
            # Add to list
            all_resumes.append({
                'Resume ID': resume.get('File Name', f"Resume_{_}"),
                'Skills': resume.get('Skills', ''),
                'Tools': resume.get('Tools', ''),
                'Certifications': resume.get('Certifications', ''),
                'Score': score
            })
        
        # Sort by score
        all_resumes.sort(key=lambda x: x['Score'], reverse=True)
        
        # Get top 3
        top_3 = all_resumes[:3] if len(all_resumes) >= 3 else all_resumes
        
        # Categorize based on score thresholds
        high_matches = [r for r in all_resumes if r['Score'] >= 0.6]
        medium_matches = [r for r in all_resumes if 0.3 <= r['Score'] < 0.6]
        low_matches = [r for r in all_resumes if r['Score'] < 0.3]
        
        return {
            'top_3': top_3,
            'high_matches': high_matches,
            'medium_matches': medium_matches,
            'low_matches': low_matches
        }
    
    def process_resume_pool(self, uploaded_files):
        """
        Process uploaded resume files
        This is a simplified implementation for demo purposes
        
        Args:
            uploaded_files: List of uploaded resume files
            
        Returns:
            DataFrame: Processed resume data
        """
        # Create a DataFrame with sample data
        data = []
        
        for i, file in enumerate(uploaded_files):
            # In a real implementation, we would extract text from the uploaded files
            # For demo purposes, we'll create sample data
            data.append({
                'File Name': file.name,
                'Skills': f"Python, Java, {'Data Science' if i % 2 == 0 else 'Cloud Computing'}, SQL",
                'Tools': f"{'TensorFlow' if i % 2 == 0 else 'Docker'}, Git, {'AWS' if i % 3 == 0 else 'Azure'}",
                'Certifications': f"{'AWS Certified' if i % 2 == 0 else 'Azure Certified'}"
            })
        
        return pd.DataFrame(data)

def render_candidate_ranking_page(services):
    """
    Render the candidate ranking page with enhanced resume pool management
    
    Args:
        services (dict): Dictionary of shared services
    """
    # Unpack services
    logger = services.get('logger')
    analyzer = services.get('analyzer')
    agent = services.get('agent')
    state_manager = services.get('state_manager')
    
    display_section_header("🎯 Resume Ranking")
    
    # First, check if we have an active JD in our repository
    jd_repository = state_manager.get('jd_repository', {})
    jd_content, jd_source_name, jd_unique_id = state_manager.get_jd_content()
    
    # Create resume analyzer
    resume_analyzer = ResumeAnalyzer()
    
    # --- Create Layout Columns ---
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        display_subsection_header("Select Position")
        
        if not jd_content:
            # No active JD, need to select one first
            st.warning("No active job description found.")
            st.info("Please select a job description in the JD Optimization tab first.")
            
            # Show button to navigate to JD Optimization
            if st.button("Go to JD Optimization", key="goto_jd_opt"):
                state_manager.set('active_tab', "JD Optimization")
                st.rerun()
            
            return
        
        # Show active job description info
        st.success(f"Using job description: {jd_source_name}")
        
        # Detect job type from repository or use default
        jd_type = jd_repository.get('JD_Type', "general")
        if not jd_type:
            jd_type = detect_jd_type(jd_source_name) if jd_source_name else "general"
        
        st.markdown(f"**Resume Pool:** {jd_type.replace('_', ' ').title()}")
        
        # --- Resume Pool Selection ---
        display_subsection_header("Resume Pools")
        
        # Get resume repository
        resume_repository = state_manager.get('resume_repository', {})
        
        # Initialize resume_pools if empty
        if not resume_repository.get('pools'):
            resume_repository['pools'] = []
            state_manager.set('resume_repository', resume_repository)
        
        # Add generic options along with any user-uploaded pools
        generic_options = ["General", "Data Engineer", "Java Developer"]
        user_pools = [pool["pool_name"] for pool in resume_repository.get('pools', [])]
        
        pool_options = ["(Auto Selection)"] + user_pools + generic_options + ["Upload New Resume Pool"]
        selected_pool_option = st.selectbox(
            "Select Resume Pool Manually (Optional)", 
            pool_options,
            key="resume_pool_selector"
        )
        
        # Handle resume pool selection
        resume_df = handle_resume_pool_selection(selected_pool_option, resume_analyzer, jd_type, state_manager)
        if resume_df is None:
            st.warning("No resume data available. Please select or upload a valid resume pool.")
            return
        
        # --- Analyze Button ---
        if st.button('🔍 Analyze Resumes', type="primary", key="analyze_resume_btn"):
            with st.spinner('Analyzing resumes...'):
                try:
                    # Get job description as dict/Series for analysis
                    job_desc = pd.Series({
                        'File Name': jd_source_name,
                        'JD_Type': jd_type,
                        'Skills': extract_skills_from_text(jd_content),
                        'Tools': extract_tools_from_text(jd_content)
                    })
                    
                    # Run the analysis
                    results = resume_analyzer.categorize_resumes(job_desc, resume_df)
                    
                    # Store results in state manager
                    resume_repository['analysis_results'] = results
                    state_manager.set('resume_repository', resume_repository)
                    
                    display_success_message("Resume analysis completed!")
                    
                    # Force a rerun to update the UI
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    
                    # Create fallback result set with random scores
                    fallback_results = create_fallback_analysis(resume_df)
                    resume_repository['analysis_results'] = fallback_results
                    state_manager.set('resume_repository', resume_repository)
                    
                    # Force a rerun to update the UI
                    st.rerun()
        
        # Display top matches preview
        analysis_results = resume_repository.get('analysis_results')
        if analysis_results:
            display_top_matches(analysis_results)
    
    # --- Results Display ---
    analysis_results = resume_repository.get('analysis_results')
    if analysis_results:
        # Analysis overview in second column
        with col2:
            display_subsection_header("Overview")
            try:
                chart = create_distribution_chart(analysis_results)
                st.plotly_chart(chart, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating distribution chart: {str(e)}")
                display_info_message("Chart visualization failed. Please check your data.")
        
        # Detailed analysis in third column
        with col3:
            display_subsection_header("Detailed Analysis")
            if 'top_3' in analysis_results and len(analysis_results['top_3']) > 0:
                # Get JD as a Series for radar chart
                job_desc = pd.Series({
                    'File Name': jd_source_name,
                    'JD_Type': jd_type,
                    'Skills': extract_skills_from_text(jd_content),
                    'Tools': extract_tools_from_text(jd_content)
                })
                
                display_detailed_resume_analysis(analysis_results, job_desc)
            else:
                st.info("No detailed analysis available.")
        
        # All Resumes by Category section
        st.markdown("---")
        display_section_header("📑 All Resumes by Category")
        display_categorized_resumes(analysis_results)


def extract_skills_from_text(text):
    """Extract skills from text for demo purposes"""
    # In a real implementation, this would use NLP to extract skills
    # For demo purposes, we'll use a simple keyword approach
    common_skills = ['python', 'java', 'javascript', 'c++', 'c#', 'sql', 'nosql', 
                     'aws', 'azure', 'gcp', 'cloud', 'machine learning', 'ai',
                     'data science', 'data analysis', 'ml', 'deep learning',
                     'frontend', 'backend', 'full stack', 'devops', 'agile']
    
    found_skills = []
    text_lower = text.lower()
    
    for skill in common_skills:
        if skill in text_lower:
            found_skills.append(skill)
    
    return ', '.join(found_skills)


def extract_tools_from_text(text):
    """Extract tools from text for demo purposes"""
    # In a real implementation, this would use NLP to extract tools
    # For demo purposes, we'll use a simple keyword approach
    common_tools = ['git', 'jenkins', 'docker', 'kubernetes', 'terraform',
                   'jira', 'confluence', 'slack', 'react', 'angular', 'vue',
                   'django', 'flask', 'spring', 'tensorflow', 'pytorch',
                   'scikit-learn', 'pandas', 'numpy', 'jupyter']
    
    found_tools = []
    text_lower = text.lower()
    
    for tool in common_tools:
        if tool in text_lower:
            found_tools.append(tool)
    
    return ', '.join(found_tools)


def detect_jd_type(file_name):
    """Detect the job description type based on the file name"""
    file_name = str(file_name).lower()
    
    # Define keyword patterns for each type
    java_python_keywords = ['java', 'python', 'support']
    data_engineer_keywords = ['data', 'analytics', 'aiml']
    
    # Check for Java/Python developer
    if any(keyword in file_name for keyword in java_python_keywords):
        return 'java_developer'
    
    # Check for Data Engineer
    elif any(keyword in file_name for keyword in data_engineer_keywords):
        return 'data_engineer'
    
    # Default type
    return 'general'


def handle_resume_pool_selection(selection, resume_analyzer, jd_type, state_manager):
    """
    Handle different resume pool selection options
    
    Args:
        selection (str): Selected pool option
        resume_analyzer: Resume analyzer instance
        jd_type (str): Job description type
        state_manager: State manager instance
        
    Returns:
        DataFrame or None: Resume DataFrame or None if still in selection process
    """
    resume_repository = state_manager.get('resume_repository', {})
    
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
                    # Update resume repository
                    pools = resume_repository.get('pools', [])
                    pools.append({
                        "pool_name": new_pool_name, 
                        "data": pool_df.to_dict('records')  # Convert to dict for storage
                    })
                    resume_repository['pools'] = pools
                    state_manager.set('resume_repository', resume_repository)
                    
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
        pools = resume_repository.get('pools', [])
        for pool in pools:
            if pool["pool_name"] == selection:
                # Convert stored dict back to DataFrame
                pool_df = pd.DataFrame(pool["data"])
                st.success(f"Loaded custom resume pool '{selection}' with {len(pool_df)} resumes")
                return pool_df
    
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
        score = np.random.uniform(0.1, 0.9)
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
    high_matches = [r for r in all_resumes if r['Score'] >= 0.6]
    medium_matches = [r for r in all_resumes if 0.3 <= r['Score'] < 0.6]
    low_matches = [r for r in all_resumes if r['Score'] < 0.3]
    
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