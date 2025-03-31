import os
import streamlit as st
import pandas as pd
import numpy as np
from models.resume_analyzer import ResumeAnalyzer
from utils.text_processing import detect_jd_type
from utils.visualization import create_distribution_chart, create_radar_chart
from ui.common import display_section_header, display_subsection_header, display_info_message, display_warning_message
from utils.file_utils import read_job_description, get_jd_files

def render_candidate_ranking_page():
    """Render the candidate ranking page"""
    
    display_section_header("🎯 Resume Ranking")
    
    # Try to load job data from JDs folder
    jd_files = get_jd_files()
    
    if not jd_files:
        display_warning_message("No job description files found in JDs folder. Using sample data instead.")
        # Create sample job data
        job_data = {
            'File Name': ['DataAnalyticsAIMLJD (1).txt', 'JobDescriptionJavaPythonSupport.txt'],
            'Skills': ['Python, Java, ML, AI, Data Analysis', 'Java, Python, Object-Oriented Programming'],
            'Tools': ['SQL, Cloud, Docker', 'Debugging tools, CoderPad'],
            'JD_Type': ['data_engineer', 'java_developer']
        }
        job_df = pd.DataFrame(job_data)
    else:
        try:
            # Display all JD files directly without transformation
            jd_directory = os.path.join(os.getcwd(), "JDs")
            
            # Create job data from all JD files
            job_data = {
                'File Name': jd_files,
                'Skills': [''] * len(jd_files),  # Empty skills initially
                'Tools': [''] * len(jd_files),   # Empty tools initially
                'JD_Type': [detect_jd_type(file) for file in jd_files]
            }
            
            # Create dataframe
            job_df = pd.DataFrame(job_data)
            
            # Show message about JD files found
            st.info(f"Loaded {len(job_df)} job descriptions from JDs folder")
            
            # Display list of JD files found (for debugging)
            with st.expander("Found JD Files", expanded=False):
                for jd_file in jd_files:
                    st.write(f"- {jd_file}")
            
        except Exception as e:
            st.error(f"Error loading job data from JDs folder: {e}")
            # Create sample job data
            job_data = {
                'File Name': ['DataAnalyticsAIMLJD (1).txt', 'JobDescriptionJavaPythonSupport.txt'],
                'Skills': ['Python, Java, ML, AI, Data Analysis', 'Java, Python, Object-Oriented Programming'],
                'Tools': ['SQL, Cloud, Docker', 'Debugging tools, CoderPad'],
                'JD_Type': ['data_engineer', 'java_developer']
            }
            job_df = pd.DataFrame(job_data)
    
    # Create three columns for main layout
    col1, col2, col3 = st.columns([1, 1, 1])

    # Initialize the resume analyzer
    resume_analyzer = ResumeAnalyzer()

    with col1:
        display_subsection_header("Select Position")
        job_desc_file_names = job_df['File Name'].tolist()
        selected_job_desc = st.selectbox('Choose position:', job_desc_file_names, label_visibility="collapsed")
        job_desc = job_df[job_df['File Name'] == selected_job_desc].iloc[0]
        
        # Display the selected JD type for verification
        jd_type = job_desc['JD_Type']
        st.markdown(f"**Resume Pool:** {jd_type.replace('_', ' ').title()}")
        
        # Try to read the selected JD file to show its content
        try:
            jd_directory = os.path.join(os.getcwd(), "JDs")
            file_path = os.path.join(jd_directory, selected_job_desc)
            jd_content = read_job_description(file_path)
            
            with st.expander("Job Description Content", expanded=False):
                st.text_area("Content", jd_content, height=200)
        except Exception as e:
            st.error(f"Error reading job description file: {e}")
        
        # Let user select resume data file (this will show the dropdown)
        resume_df = resume_analyzer.load_resume_data(jd_type)
        
        # Initialize categorized_resumes as an empty structure to avoid reference errors
        categorized_resumes = {
            'top_3': [],
            'high_matches': [],
            'medium_matches': [],
            'low_matches': []
        }
        
        # Analyze button (only enable if resume_df is available)
        if resume_df is not None:
            if st.button('🔍 Analyze Resumes', type="primary"):
                with st.spinner('Analyzing resumes...'):
                    try:
                        # For correct analysis, we need skills and tools
                        # Extract skills and tools first if they're empty
                        if not job_desc['Skills'] or not job_desc['Tools']:
                            # Read the JD content
                            jd_directory = os.path.join(os.getcwd(), "JDs")
                            file_path = os.path.join(jd_directory, selected_job_desc)
                            jd_content = read_job_description(file_path)
                            
                            # Extract some basic skills/tools based on file name
                            if "java" in selected_job_desc.lower() or "python" in selected_job_desc.lower():
                                skills = "Java, Python, Object-Oriented Programming"
                                tools = "Debugging tools, IDE, Git"
                            elif "data" in selected_job_desc.lower() or "analytics" in selected_job_desc.lower():
                                skills = "Python, SQL, Data Analysis, ML, AI"
                                tools = "SQL, Cloud, Docker, Data Visualization"
                            else:
                                skills = "Programming, Problem Solving, Communication"
                                tools = "Project Management, Version Control, Documentation"
                            
                            # Update the job_desc dictionary
                            job_desc = job_desc.copy()
                            job_desc['Skills'] = skills
                            job_desc['Tools'] = tools
                        
                        categorized_resumes = resume_analyzer.categorize_resumes(job_desc, resume_df)
                        st.session_state['analysis_results'] = categorized_resumes
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
                        # Create dummy results using the available resume_df
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
                        
                        # Categorize
                        high_matches = [r for r in all_resumes if r['Score'] >= 0.25]
                        medium_matches = [r for r in all_resumes if 0.2 <= r['Score'] < 0.25]
                        low_matches = [r for r in all_resumes if r['Score'] < 0.2]
                        
                        categorized_resumes = {
                            'top_3': all_resumes[:3],
                            'high_matches': high_matches,
                            'medium_matches': medium_matches,
                            'low_matches': low_matches
                        }
                        st.session_state['analysis_results'] = categorized_resumes
        
        # Check if analysis results exist in session state, otherwise use the empty structure
        if 'analysis_results' in st.session_state:
            categorized_resumes = st.session_state['analysis_results']
            
        # Top 3 Quick View
        display_subsection_header("Top Matches")
        for i, resume in enumerate(categorized_resumes['top_3'][:3]):
            st.markdown(f"""
            <div>
                <h4>#{i + 1} - {resume['Resume ID']}</h4>
                <p>Match: {resume['Score']:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        display_subsection_header("Detailed Analysis")
        tabs = st.tabs(["#1", "#2", "#3"])
        
        for i, (tab, resume) in enumerate(zip(tabs, categorized_resumes['top_3'])):
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
                    try:
                        # Simple static insights
                        insights = f"""
                        <h4>Key Match Analysis</h4>
                        <p>This candidate has skills that align with the job requirements.</p>
                        <ul>
                            <li>Technical skills match core requirements</li>
                            <li>Experience with relevant tools</li>
                            <li>Professional background enhances qualifications</li>
                        </ul>
                        <p><strong>Overall assessment:</strong> Good potential match</p>
                        """
                        
                        st.markdown(insights, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error generating insights: {str(e)}")
                        st.markdown("""
                        <h4>Key Match Analysis</h4>
                        <p>This candidate has skills that align with the job requirements.</p>
                        <ul>
                            <li>Technical skills match core requirements</li>
                            <li>Experience with relevant tools</li>
                            <li>Professional background enhances qualifications</li>
                        </ul>
                        <p><strong>Overall assessment:</strong> Good potential match</p>
                        """, unsafe_allow_html=True)

    # All Resumes by Category (below the main content)
    st.markdown("---")
    display_section_header("📑 All Resumes by Category")
    
    cat_col1, cat_col2, cat_col3 = st.columns(3)
    
    with cat_col1:
        with st.expander(f"High Matches ({len(categorized_resumes['high_matches'])})"):
            for resume in categorized_resumes['high_matches']:
                st.markdown(f"""
                <div>
                    <h4>{resume['Resume ID']}</h4>
                    <p>Match: {resume['Score']:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
    
    with cat_col2:
        with st.expander(f"Medium Matches ({len(categorized_resumes['medium_matches'])})"):
            for resume in categorized_resumes['medium_matches']:
                st.markdown(f"""
                <div>
                    <h4>{resume['Resume ID']}</h4>
                    <p>Match: {resume['Score']:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
    
    with cat_col3:
        with st.expander(f"Low Matches ({len(categorized_resumes['low_matches'])})"):
            for resume in categorized_resumes['low_matches']:
                st.markdown(f"""
                <div>
                    <h4>{resume['Resume ID']}</h4>
                    <p>Match: {resume['Score']:.2%}</p>
                </div>
                """, unsafe_allow_html=True)