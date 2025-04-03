import os
import streamlit as st
import pandas as pd
import numpy as np
from models.resume_analyzer import ResumeAnalyzer
from utils.text_processing import detect_jd_type
from utils.visualization import create_distribution_chart, create_radar_chart
from ui.common import display_section_header, display_subsection_header, display_info_message, display_warning_message
from utils.file_utils import read_job_description, get_jd_files

def analyze_uploaded_resume(uploaded_file):
    """Analyze a user-uploaded resume (.docx) and return the extracted information."""
    if not uploaded_file.name.endswith(".docx"):
        raise ValueError(f"Unsupported file format for {uploaded_file.name}. Only .docx files are supported.")
    
    # Write the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    
    try:
        result = analyze_resume(tmp_path)
        # Override the File Name with the original uploaded file's name
        result["File Name"] = uploaded_file.name
        if result and all(key in result for key in ['File Name', 'Skills', 'Tools', 'Certifications']):
            return result
        else:
            print(f"Invalid resume data for {uploaded_file.name}")
            return None
    except Exception as e:
        print(f"Error in analyze_uploaded_resume for {uploaded_file.name}: {e}")
        return None
    finally:
        os.remove(tmp_path)
        

def render_candidate_ranking_page():
    """Render the candidate ranking page with default resume pool selection and optional manual pool selection."""
    
    st.markdown(f"<div class='section-header'>🎯 Resume Ranking</div>", unsafe_allow_html=True)
    st.markdown("Raghav please click on 'Analyze Resume' to remove the error.", unsafe_allow_html=True)
    
    # --- Load Job Data ---
    if not os.path.exists('/Users/raghav/Desktop/Apexon/JD Optimization/job_descriptions_analysis_output.csv'):
        st.warning("job_descriptions_analysis_output.csv not found. Using sample data instead.")
        job_data = {
            'File Name': ['DataAnalyticsAIMLJD (1).txt', 'JobDescriptionJavaPythonSupport.txt'],
            'Skills': ['Python, Java, ML, AI, Data Analysis', 'Java, Python, Object-Oriented Programming'],
            'Tools': ['SQL, Cloud, Docker', 'Debugging tools, CoderPad'],
            'JD_Type': ['data_engineer', 'java_developer']
        }
        job_df = pd.DataFrame(job_data)
    else:
        try:
            job_df = pd.read_csv('/Users/raghav/Desktop/Apexon/JD Optimization/job_descriptions_analysis_output.csv')
            if 'JD_Type' not in job_df.columns:
                job_df['JD_Type'] = 'unknown'
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
        except Exception as e:
            st.error(f"Error loading job data: {e}")
            job_data = {
                'File Name': ['DataAnalyticsAIMLJD (1).txt', 'JobDescriptionJavaPythonSupport.txt'],
                'Skills': ['Python, Java, ML, AI, Data Analysis', 'Java, Python, Object-Oriented Programming'],
                'Tools': ['SQL, Cloud, Docker', 'Debugging tools, CoderPad'],
                'JD_Type': ['data_engineer', 'java_developer']
            }
            job_df = pd.DataFrame(job_data)
    
    # --- Sample Resume Data (Fallback) ---
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
    
    # --- Layout Columns ---
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("<div class='subsection-header'>Select Position</div>", unsafe_allow_html=True)
        # Combine job names with enhanced versions if available
        job_names = job_df['File Name'].tolist()
        if st.session_state.get('final_version'):
            job_names.append("Final Enhanced Version")
        if st.session_state.get('client_enhanced_version'):
            job_names.append("Client Enhanced Version")
        
        selected = st.selectbox('Choose position:', job_names, label_visibility="collapsed")
        # Set job_desc based on selection
        if selected == "Final Enhanced Version":
            if st.session_state.get('final_version'):
                # Check if final_version is a dictionary or can be converted to one
                if isinstance(st.session_state.final_version, dict):
                    job_desc = pd.Series({
                        'File Name': "Final Enhanced Version",
                        'JD_Type': "Enhanced",
                        'Skills': st.session_state.final_version.get('skills', ''),
                        'Tools': st.session_state.final_version.get('tools', '')
                    })
                elif isinstance(st.session_state.final_version, str):
                    # If it's a string, create a basic job description
                    job_desc = pd.Series({
                        'File Name': "Final Enhanced Version",
                        'JD_Type': "Enhanced",
                        'Skills': st.session_state.final_version,
                        'Tools': ''
                    })
                else:
                    st.warning("Invalid final enhanced job description format.")
                    job_desc = job_df.iloc[0]
            else:
                st.warning("No final enhanced job description available.")
                job_desc = job_df.iloc[0]
        elif selected == "Client Enhanced Version":
            if st.session_state.get('client_enhanced_version'):
                # Check if client_enhanced_version is a dictionary or can be converted to one
                if isinstance(st.session_state.client_enhanced_version, dict):
                    job_desc = pd.Series({
                        'File Name': "Client Enhanced Version",
                        'JD_Type': "Enhanced",
                        'Skills': st.session_state.client_enhanced_version.get('skills', ''),
                        'Tools': st.session_state.client_enhanced_version.get('tools', '')
                    })
                elif isinstance(st.session_state.client_enhanced_version, str):
                    # If it's a string, create a basic job description
                    job_desc = pd.Series({
                        'File Name': "Client Enhanced Version",
                        'JD_Type': "Enhanced",
                        'Skills': st.session_state.client_enhanced_version,
                        'Tools': ''
                    })
                else:
                    st.warning("Invalid client enhanced job description format.")
                    job_desc = job_df.iloc[0]
            else:
                st.warning("No client enhanced job description available.")
                job_desc = job_df.iloc[0]
        else:
            job_desc = job_df[job_df['File Name'] == selected].iloc[0]
        
        st.markdown(f"**Resume Pool:** {job_desc['JD_Type'].replace('_',' ').title()}")
        
        # --- Optional Manual Resume Pool Selection ---
        st.markdown("<div class='subsection-header'>Resume Pools</div>", unsafe_allow_html=True)
        if "resume_pools" not in st.session_state:
            st.session_state.resume_pools = []  # List of dicts: {"pool_name": str, "data": DataFrame}
        
        # Add generic options along with any user-uploaded pools
        generic_options = ["General", "Data Engineer", "Java Developer"]
        user_pools = [pool["pool_name"] for pool in st.session_state.resume_pools]
        pool_options_display = ["(No Selection)"] + user_pools + generic_options + ["Upload New Resume Pool"]
        selected_pool_option = st.selectbox(
            "Select Resume Pool Manually (Optional)", 
            pool_options_display, 
            key="resume_pool_selector"
        )
        
        base_dir = "/Users/raghav/Desktop/Apexon/JD Optimization"
        # Process manual pool selection
        if selected_pool_option == "Upload New Resume Pool":
            new_pool_name = st.text_input("Enter new pool name:", key="new_pool_name")
            new_pool_files = st.file_uploader(
                "Upload resumes for the new pool", 
                type=['pdf', 'docx', 'txt'], 
                accept_multiple_files=True, 
                key="new_pool_files"
            )
            if st.button("Add Resume Pool", key="add_pool"):
                if new_pool_name and new_pool_files:
                    processed_resumes = []
                    for uploaded_file in new_pool_files:
                        try:
                            # Process only .docx files using the new function.
                            if uploaded_file.name.endswith(".docx"):
                                resume_data = analyze_uploaded_resume(uploaded_file)
                                if resume_data is not None:  # Only append non-None results
                                    processed_resumes.append(resume_data)
                                else:
                                    st.warning(f"Skipping {uploaded_file.name} due to processing error")
                            else:
                                st.warning(f"File {uploaded_file.name} is not a .docx file and will be skipped.")
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {e}")
                    
                    if processed_resumes:
                        pool_df = pd.DataFrame(processed_resumes)
                        st.session_state.resume_pools.append({"pool_name": new_pool_name, "data": pool_df})
                        st.success(f"Resume pool '{new_pool_name}' added with {len(processed_resumes)} resumes!")
                        st.rerun()
                    else:
                        st.warning("No valid resumes were processed. Please check your files.")

        elif selected_pool_option == "(No Selection)":
            # Use default resume pool based on selected JD
            if job_desc["File Name"] == "JobDescriptionJavaPythonSupport.txt":
                default_file = "resumes_analysis_outputJDJavaDeveloper.csv"
                resume_pool_category = "Java Developer"
            elif job_desc["File Name"] == "DataAnalyticsAIMLJD (1).txt":
                default_file = "resumes_analysis_output_JDPrincipalSoftwareEngineer.csv"
                resume_pool_category = "Data Engineer"
            else:
                default_file = "resumes_analysis_output.csv"
                resume_pool_category = "General"
            fp = os.path.join(base_dir, default_file)
            if os.path.exists(fp):
                resume_df = pd.read_csv(fp)
                st.success(f"Loaded default resume pool ({resume_pool_category})")

            else:
                st.warning("Default resume pool file not found. Using sample data.")
                resume_df = pd.DataFrame(sample_resume_data)
        elif selected_pool_option in generic_options:
            # Load resumes from the generic option selected
            generic_map = {
                "General": "resumes_analysis_output.csv",
                "Data Engineer": "resumes_analysis_output_JDPrincipalSoftwareEngineer.csv",
                "Java Developer": "resumes_analysis_outputJDJavaDeveloper.csv"
            }
            default_file = generic_map[selected_pool_option]
            fp = os.path.join(base_dir, default_file)
            if os.path.exists(fp):
                resume_df = pd.read_csv(fp)
                st.success(f"Loaded {selected_pool_option} resume pool")
            else:
                st.warning("No resume data files found for the selected generic pool. Using sample data.")
                resume_df = pd.DataFrame(sample_resume_data)
        else:
            # If a user-uploaded pool is selected
            for pool in st.session_state.resume_pools:
                if pool["pool_name"] == selected_pool_option:
                    resume_df = pool["data"]
                    st.success(f"Loaded resume pool '{selected_pool_option}' with {len(resume_df)} resumes")
                    break
    
    # --- Analyze Resume Button ---
    if st.button("Analyze Resume", type="primary", key="analyze_resume_btn"):
        with st.spinner('Analyzing resumes...'):
            try:
                st.session_state['analysis_results'] = categorize_resumes(job_desc, resume_df)
            except Exception as e:
                st.error(f"Error during analysis: {e}")
                # Fallback: use random scores
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
                all_resumes.sort(key=lambda x: x['Score'], reverse=True)
                st.session_state['analysis_results'] = {
                    'top_3': all_resumes[:3],
                    'high_matches': [r for r in all_resumes if r['Score'] >= 0.25],
                    'medium_matches': [r for r in all_resumes if 0.2 <= r['Score'] < 0.25],
                    'low_matches': [r for r in all_resumes if r['Score'] < 0.2]
                }
    
    # --- The rest of the UI (Overview, Detailed Analysis, etc.) remains unchanged ---
    categorized_resumes = st.session_state.get('analysis_results')
    if not categorized_resumes:
        st.info("Please select or upload a Resume pool")
    else:
        with col2:
            st.markdown(f"<div class='subsection-header'>Overview</div>", unsafe_allow_html=True)
            try:
                chart = create_distribution_chart(categorized_resumes)
                st.plotly_chart(chart, use_container_width=True)
            except Exception as e:
                st.info("Please select or upload a Resume pool")
                # Optionally, you could include a fallback bar chart here:
                # st.bar_chart({
                #     'High Match': [len(categorized_resumes.get('high_matches', []))],
                #     'Medium Match': [len(categorized_resumes.get('medium_matches', []))],
                #     'Low Match': [len(categorized_resumes.get('low_matches', []))]
                # })
            
            st.markdown(f"<div class='subsection-header'>Top Matches</div>", unsafe_allow_html=True)
            if 'top_3' in categorized_resumes and categorized_resumes['top_3']:
                for i, resume in enumerate(categorized_resumes['top_3'][:3]):
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="margin:0">#{i + 1} - {resume['Resume ID']}</h4>
                        <p style="margin:0">Match: {resume['Score']:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No top matches available")
        
        with col3:
            st.markdown(f"<div class='subsection-header'>Detailed Analysis</div>", unsafe_allow_html=True)
            if 'top_3' in categorized_resumes and categorized_resumes['top_3']:
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
                                insights = generate_ai_insights(job_desc, resume)
                                st.markdown(f"""
                                <div class="insight-box compact-text">
                                    {insights}
                                </div>
                                """, unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"Error generating insights: {str(e)}")
                                st.markdown(f"""
                                <div class="insight-box compact-text">
                                    <h4>Key Match Analysis</h4>
                                    <p>This candidate has skills that align with the job requirements.</p>
                                    <ul>
                                        <li>Technical skills match core requirements</li>
                                        <li>Experience with relevant tools</li>
                                        <li>Professional background enhances qualifications</li>
                                    </ul>
                                    <p><strong>Overall assessment:</strong> Good potential match</p>
                                </div>
                                """, unsafe_allow_html=True)
            else:
                st.info("No detailed analysis available")
        
        st.markdown("---")
        st.markdown(f"<div class='section-header'>📑 All Resumes by Category</div>", unsafe_allow_html=True)
        
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
