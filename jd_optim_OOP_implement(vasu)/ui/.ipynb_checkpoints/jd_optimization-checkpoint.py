import os
import streamlit as st
import datetime
import json
import pandas as pd
from utils.file_utils import read_job_description, save_enhanced_jd
from utils.visualization import create_multi_radar_chart, create_comparison_dataframe
from ui.common import (
    display_section_header, display_subsection_header, 
    display_warning_message, display_info_message, display_success_message
)
from utils.job_search import render_job_search_section, find_data_files

def render_jd_optimization_page(services):
    """
    Render the unified JD Optimization page with optimized performance by using cached results
    
    Args:
        services (dict): Dictionary of shared services 
    """
    # Unpack services
    logger = services.get('logger')
    analyzer = services.get('analyzer')
    agent = services.get('agent')
    state_manager = services.get('state_manager')
    
    display_section_header("📝 JD Optimization")
    
    # Create tabs for the main content and feedback history
    main_tabs = st.tabs(["JD Optimization", "Feedback History"])
    
    with main_tabs[0]:
        ##########################
        # Part 1: Unified Job Description Selection
        ##########################
        display_subsection_header("1. Select Job Description")
        
        # Create a selection for the source type
        source_options = ["📁 File Selection", "📤 Upload New", "🔍 Search Database"]
        selected_source = st.radio(
            "Choose job description source:",
            options=source_options,
            horizontal=True,
            key="jd_source_selector"
        )
        
        # Variables to track job description source
        jd_content = None
        jd_source_name = None
        jd_unique_id = None
        
        # Handle each source option
        if selected_source == "🔍 Search Database":
            # Initialize job search utility if needed
            job_search_initialized = render_job_search_section(state_manager)
            
            if not job_search_initialized:
                st.warning("Please initialize the job search database first using the options above.")
                return
                
            job_search = state_manager.get('job_search_utility')
            
            # Check if the job search has been initialized
            if not job_search.is_initialized:
                # Display warning message about not being initialized
                st.warning("Job search database not properly initialized. Please try reloading the page.")
                return
            else:
                # Get dropdown options
                options = job_search.get_dropdown_options()
                
                if not options:
                    st.warning("No job listings found in the data.")
                else:
                    # Add a search box for filtering the dropdown
                    st.markdown("**💼 Search for a job description:**")
                    search_col1, search_col2 = st.columns([3, 1])
                    
                    with search_col1:
                        search_term = st.text_input("Search for job by ID, name, or client:", key="job_search_term")
                    
                    with search_col2:
                        st.markdown("""
                        <span title="Search Help">ℹ️ Search by ID, name, or client</span>
                        """, unsafe_allow_html=True)
                    
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
                            key="job_search_dropdown"
                        )
                        
                        # Find and display the job description
                        if selected_option:
                            job_description, job_details = job_search.find_job_description(selected_option)
                            
                            if job_description:
                                jd_content = job_description
                                jd_source_name = selected_option
                                jd_unique_id = f"db_{job_details.get('Job Id', '')}"
                                
                                # Display job details in an expander
                                with st.expander("Job Details", expanded=False):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown(f"**Job ID:** {job_details.get('Job Id', 'N/A')}")
                                        st.markdown(f"**Reference ID:** {job_details.get('Reference Id', 'N/A')}")
                                    with col2:
                                        st.markdown(f"**Parent ID:** {job_details.get('Parent Id', 'N/A')}")
                                        st.markdown(f"**ATS Position ID:** {job_details.get('ATS Position ID', 'N/A')}")
                            else:
                                st.error("Could not find job description for the selected job.")
        
        elif selected_source == "📁 File Selection":
            jd_directory = os.path.join(os.getcwd(), "Data", "JDs")
            try:
                # Create directory if it doesn't exist
                if not os.path.exists(jd_directory):
                    os.makedirs(jd_directory, exist_ok=True)
                    st.info("Created Data/JDs directory.")
                
                files = [f for f in os.listdir(jd_directory) if f.endswith(('.txt', '.docx'))]
                
                if files:
                    selected_file = st.selectbox(
                        "Select Job Description File", 
                        files, 
                        key="file_selector"
                    )
                    
                    if selected_file:
                        # Load selected file
                        file_path = os.path.join(jd_directory, selected_file)
                        
                        try:
                            file_content = read_job_description(file_path)
                            jd_content = file_content
                            jd_source_name = selected_file
                            jd_unique_id = f"file_{selected_file}"
                        except Exception as e:
                            st.error(f"Error reading file: {str(e)}")
                else:
                    st.info("No job description files found in the Data/JDs directory. Please upload one or create a new file.")
            except Exception as e:
                st.error(f"Error accessing JDs directory: {str(e)}")
        
        elif selected_source == "📤 Upload New":
            uploaded_file = st.file_uploader(
                "Upload Job Description File", 
                type=['txt', 'docx'],
                key="file_uploader"
            )
            
            if uploaded_file:
                # Process uploaded file
                try:
                    if uploaded_file.name.endswith('.txt'):
                        file_content = uploaded_file.getvalue().decode('utf-8')
                    else:  # .docx
                        try:
                            from docx import Document
                            # Save temporarily
                            temp_path = f"temp_{uploaded_file.name}"
                            with open(temp_path, 'wb') as f:
                                f.write(uploaded_file.getvalue())
                            
                            # Read with python-docx
                            doc = Document(temp_path)
                            file_content = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
                            
                            # Clean up
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                        except ImportError:
                            st.error("python-docx package not found. Please install it to process DOCX files.")
                            file_content = f"[Could not process DOCX file: {uploaded_file.name}]"
                    
                    jd_content = file_content
                    jd_source_name = uploaded_file.name
                    jd_unique_id = f"upload_{uploaded_file.name}"
                    
                    # Save to JDs directory for future use
                    jd_dir = os.path.join(os.getcwd(), "Data", "JDs")
                    os.makedirs(jd_dir, exist_ok=True)
                    save_path = os.path.join(jd_dir, uploaded_file.name)
                    
                    with open(save_path, 'wb') as f:
                        f.write(uploaded_file.getvalue())
                    
                    st.success(f"Saved {uploaded_file.name} to Data/JDs directory for future use.")
                except Exception as e:
                    st.error(f"Error processing uploaded file: {str(e)}")
        
        # Use the job description content if available
        if jd_content:
            # Store in session state
            state_manager.handle_jd_selection(jd_content, jd_source_name, jd_unique_id, "jd_optimization")
            
            # Display the job description preview
            st.markdown("<div class='subsection-header'>Job Description Preview</div>", unsafe_allow_html=True)
            
            # Show the JD preview in a collapsible section
            with st.expander("View Job Description", expanded=True):
                st.text_area(
                    "Content", 
                    jd_content, 
                    height=250, 
                    disabled=True,
                    key="jd_preview"
                )
        else:
            if selected_source:
                st.warning("Please select or upload a job description to continue.")
            return
        
        ##########################
        # Part 2: Generate Enhanced Versions (with caching)
        ##########################
        display_subsection_header("2. Generate Enhanced Versions")
        
        # Check if we already have cached versions for this JD
        jd_repository = state_manager.get('jd_repository', {})
        cached_versions = jd_repository.get('enhanced_versions', [])
        
        # Display cache status
        if cached_versions:
            st.success("📋 Found cached enhanced versions for this job description!")
            
            # Add option to regenerate if needed
            regenerate = st.checkbox("Regenerate versions (not recommended unless necessary)", value=False)
            
            if regenerate:
                generate_btn = st.button(
                    "Generate New Versions", 
                    type="primary", 
                    key="generate_btn",
                    help="Generate three new AI-enhanced versions (ignores cache)"
                )
            else:
                # Just load cached versions without an API call
                if 'enhanced_versions' not in jd_repository or jd_repository.get('reload_flag', False):
                    state_manager.update_jd_repository('reload_flag', False, source_tab="jd_optimization")
                    st.rerun()
                
                # Show fake button
                st.button(
                    "✅ Enhanced Versions Loaded", 
                    type="secondary", 
                    disabled=True,
                    key="loaded_btn"
                )
                
                # Set generate_btn to False
                generate_btn = False
        else:
            # No cached versions, show normal generate button
            generate_btn = st.button(
                "Generate Enhanced Versions", 
                type="primary", 
                key="generate_btn",
                help="Generate three AI-enhanced versions of your job description"
            )
        
        # Handle generating enhanced versions
        if generate_btn:
            with st.spinner("Generating enhanced versions... This may take a moment"):
                original_jd = jd_repository.get('original')
                if not original_jd:
                    st.error("Original job description not found. Please select a job description first.")
                    return
                
                try:
                    # Generate enhanced versions using the agent
                    if agent and hasattr(agent, 'generate_initial_descriptions'):
                        versions = agent.generate_initial_descriptions(original_jd)
                        
                        # Ensure we have 3 versions
                        while len(versions) < 3:
                            versions.append(f"Enhanced Version {len(versions)+1}:\n{original_jd}")
                        
                        # Update the repository
                        state_manager.update_jd_repository('enhanced_versions', versions, source_tab="jd_optimization")
                        
                        # Log versions generated if logger is available
                        if logger:
                            logger.log_versions_generated(versions)
                        
                        # Force rerun to show new versions
                        st.rerun()
                    else:
                        st.error("AI agent not available or missing required method.")
                except Exception as e:
                    st.error(f"Error generating enhanced versions: {str(e)}")
        
        # Display enhanced versions if available
        if cached_versions:
            # Create tabs for content and analysis
            enhanced_tabs = st.tabs(["Enhanced Versions", "Analysis & Comparison"])
            
            # Show enhanced versions tab content
            with enhanced_tabs[0]:
                version_tabs = st.tabs(["Version 1", "Version 2", "Version 3"])
                for idx, (tab, version) in enumerate(zip(version_tabs, cached_versions)):
                    with tab:
                        st.text_area(
                            f"Enhanced Version {idx + 1}",
                            version,
                            height=300,
                            disabled=True,
                            key=f"enhanced_version_{idx}"
                        )
                        
                        # Add download button for each version
                        st.download_button(
                            label=f"Download Version {idx + 1}",
                            data=version,
                            file_name=f"enhanced_jd_version_{idx+1}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            key=f"download_version_{idx}"
                        )
            
            # Show analysis & comparison tab content
            with enhanced_tabs[1]:
                # Analyze all versions
                try:
                    original_scores = analyzer.analyze_text(jd_repository.get('original', ''))
                    intermediate_scores = {
                        f'Version {i+1}': analyzer.analyze_text(version)
                        for i, version in enumerate(cached_versions)
                    }
                    
                    # Combine all scores for comparison
                    all_scores = {'Original': original_scores, **intermediate_scores}
                    
                    # Update analytics repository
                    analytics_repository = state_manager.get('analytics_repository', {})
                    analytics_repository['original_scores'] = original_scores
                    analytics_repository['version_scores'] = intermediate_scores
                    state_manager.set('analytics_repository', analytics_repository)
                    
                    analysis_col1, analysis_col2 = st.columns([1, 1])
                        
                    with analysis_col1:
                        st.subheader("Skill Coverage Comparison")
                        radar_chart = create_multi_radar_chart(all_scores)
                        st.plotly_chart(radar_chart, use_container_width=True, key="intermediate_radar")
                        
                    with analysis_col2:
                        st.subheader("Detailed Analysis")
                        comparison_df = create_comparison_dataframe(all_scores)
                        st.dataframe(
                            comparison_df,
                            height=400,
                            use_container_width=True,
                            hide_index=True,
                            key="intermediate_comparison"
                        )
                        st.caption("Percentages indicate keyword coverage in each category")
                except Exception as e:
                    st.error(f"Error generating analysis: {str(e)}")
                    st.info("Could not generate comparison analysis. Please try again later.")
        
        ##########################
        # Part 3: Provide Feedback & Generate Final Version
        ##########################
        if cached_versions:
            display_subsection_header("3. Provide Feedback & Generate Final Version")
            
            # Get feedback repository
            feedback_repository = state_manager.get('feedback_repository', {})
            
            # Create two columns for the feedback input
            left_col, right_col = st.columns([1, 1])
            
            with left_col:
                # Version selection
                selected_version = st.radio(
                    "Choose the version you'd like to use as a base:",
                    ["Version 1", "Version 2", "Version 3"],
                    help="Select the version that best matches your needs for further enhancement",
                    key="version_selector"
                )
                
                selected_index = int(selected_version[-1]) - 1  # Get version index
                
                # Update selected version in repository
                state_manager.update_jd_repository('selected_version_idx', selected_index, source_tab="jd_optimization")
                
                # Display previous feedback if available
                if feedback_repository.get('history', []):
                    st.markdown("**Previous Feedback:**")
                    with st.expander("View Feedback History", expanded=False):
                        for i, feedback in enumerate(feedback_repository.get('history', []), 1):
                            feedback_text = feedback.get("feedback", "") if isinstance(feedback, dict) else feedback
                            feedback_type = feedback.get("type", "General Feedback") if isinstance(feedback, dict) else "General Feedback"
                            feedback_role = feedback.get("role", "Unknown") if isinstance(feedback, dict) else "Unknown"
                            timestamp = feedback.get("timestamp", "")
                            
                            if timestamp:
                                try:
                                    # Format the timestamp
                                    dt = datetime.datetime.fromisoformat(timestamp)
                                    timestamp = dt.strftime("%Y-%m-%d %H:%M")
                                except:
                                    pass
                            
                            st.markdown(f"**#{i} - {feedback_type}** by {feedback_role} {timestamp}")
                            st.markdown(f"> {feedback_text}")
                            st.markdown("---")
            
            with right_col:
                # Define feedback types
                feedback_types = [
                    "General Feedback", 
                    "Rejected Candidate", 
                    "Hiring Manager Feedback", 
                    "Client Feedback", 
                    "Selected Candidate", 
                    "Interview Feedback"
                ]
                
                # Select feedback type
                current_type = feedback_repository.get('current_type', "General Feedback")
                selected_feedback_type = st.selectbox(
                    "Feedback Type:",
                    options=feedback_types,
                    index=feedback_types.index(current_type) if current_type in feedback_types else 0,
                    key="feedback_type_selector"
                )
                
                # Update type in state if changed
                if selected_feedback_type != current_type:
                    state_manager.update_feedback_repository('current_type', selected_feedback_type, source_tab="jd_optimization")
                
                # Feedback input
                current_feedback = feedback_repository.get('current_feedback', '')
                user_feedback = st.text_area(
                    "Enter your suggestions for improving the selected version:",
                    value=current_feedback,
                    height=150,
                    placeholder="E.g., 'Add more emphasis on leadership skills', 'Include cloud technologies', etc.",
                    key="user_feedback",
                    help="Be specific about what you'd like to change or improve"
                )
                
                # Update current feedback in state
                if user_feedback != current_feedback:
                    state_manager.update_feedback_repository('current_feedback', user_feedback, source_tab="jd_optimization")
                
                # Add feedback button
                if st.button("➕ Add Feedback", type="secondary", key="add_feedback"):
                    if user_feedback.strip():
                        # Create feedback object
                        feedback_obj = {
                            "feedback": user_feedback,
                            "type": selected_feedback_type,
                            "role": state_manager.get('role', "Unknown"),
                            "timestamp": datetime.datetime.now().isoformat()
                        }
                        
                        # Add to feedback history
                        history = feedback_repository.get('history', [])
                        history.append(feedback_obj)
                        state_manager.update_feedback_repository('history', history, source_tab="jd_optimization")
                        
                        # Clear current feedback
                        state_manager.update_feedback_repository('current_feedback', '', source_tab="jd_optimization")
                        
                        # Log to logger if available
                        if logger:
                            logger.log_feedback(user_feedback, selected_feedback_type)
                        
                        # Display success message
                        display_success_message("Feedback added successfully!")
                        
                        # Force refresh
                        st.rerun()
                    else:
                        display_warning_message("Please enter some feedback first.")
            
            # Generate Final JD Button
            final_version = jd_repository.get('final_version')
            
            if final_version:
                # Already have a final version
                st.success("✅ Final version already generated")
                
                # Option to regenerate
                regenerate_final = st.checkbox("Generate a new final version", value=False)
                
                if not regenerate_final:
                    # Show fake button
                    st.button(
                        "✅ Final Version Ready", 
                        type="secondary", 
                        disabled=True,
                        key="final_ready_btn"
                    )
                    generate_final_btn = False
                else:
                    generate_final_btn = st.button(
                        "🚀 Generate New Final Version", 
                        type="primary", 
                        key="generate_final_jd",
                        help="Generate a new final version"
                    )
            else:
                # No final version yet
                generate_final_btn = st.button(
                    "🚀 Generate Final Enhanced Version", 
                    type="primary", 
                    key="generate_final_jd",
                    help="Create a final version incorporating all feedback"
                )
            
            if generate_final_btn:
                try:
                    with st.spinner("Enhancing job description with feedback..."):
                        # Log version selection if using logger
                        if logger:
                            logger.log_version_selection(selected_index)
                        
                        # Get base version
                        selected_version_idx = jd_repository.get('selected_version_idx', 0)
                        versions = jd_repository.get('enhanced_versions', [])
                        if not versions or selected_version_idx >= len(versions):
                            st.error("Selected version not found. Please generate enhanced versions first.")
                            return
                            
                        base_description = versions[selected_version_idx]
                        
                        # Get feedback history
                        feedback_history = feedback_repository.get('history', [])
                        
                        # Generate final JD using AI agent
                        if agent and hasattr(agent, 'generate_final_description'):
                            final_description = agent.generate_final_description(
                                base_description, feedback_history
                            )
                            
                            # Update state
                            state_manager.update_jd_repository('final_version', final_description, source_tab="jd_optimization")
                            
                            # Log to logger if available
                            if logger:
                                logger.log_enhanced_version(final_description, is_final=True)
                            
                            display_success_message("Final version generated successfully!")
                            st.rerun()
                        else:
                            st.error("AI agent not available or missing required method.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    return
            
            # Display Final Version if available
            if final_version:
                st.markdown("---")
                display_subsection_header("✅ Final Enhanced Job Description")
                
                # Display the final enhanced version
                st.text_area(
                    "Final Content", 
                    final_version, 
                    height=400, 
                    key="final_description"
                )
                
                # Compare original vs final JD with skill analysis
                display_subsection_header("📊 Final Analysis")
                
                try:
                    # Calculate scores
                    original_scores = analyzer.analyze_text(jd_repository.get('original', ''))
                    final_scores = analyzer.analyze_text(final_version)
                    
                    # Update analytics repository
                    analytics_repository = state_manager.get('analytics_repository', {})
                    analytics_repository['final_scores'] = final_scores
                    state_manager.set('analytics_repository', analytics_repository)
                    
                    # Create comparison charts
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        final_radar = create_multi_radar_chart({'Original': original_scores, 'Final': final_scores})
                        st.plotly_chart(final_radar, use_container_width=True, key="final_radar")
                    
                    with col2:
                        final_comparison_df = create_comparison_dataframe({'Original': original_scores, 'Final': final_scores})
                        st.dataframe(
                            final_comparison_df, 
                            height=400, 
                            use_container_width=True, 
                            hide_index=True, 
                            key="final_comparison"
                        )
                except Exception as e:
                    st.error(f"Error generating analysis: {str(e)}")
                    st.info("Could not generate comparison analysis. Please try again later.")
                
                # Download Final JD
                display_subsection_header("📥 Download Options")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label="Download as TXT", 
                        data=final_version, 
                        file_name=f"enhanced_jd_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", 
                        mime="text/plain", 
                        key="download_txt"
                    )
                    if logger:
                        logger.log_download("txt", f"enhanced_jd_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
                
                with col2:
                    if st.button("Download as DOCX", key="download_docx"):
                        docx_filename = f"enhanced_jd_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
                        try:
                            save_enhanced_jd(final_version, docx_filename, 'docx')
                            display_success_message(f"Saved as {docx_filename}")
                            if logger:
                                logger.log_download("docx", docx_filename)
                        except Exception as e:
                            st.error(f"Error saving as DOCX: {str(e)}")
                            st.info("Try downloading as TXT instead.")

    # Feedback History Tab Content
    with main_tabs[1]:
        display_feedback_history_tab(state_manager, services)

def display_feedback_history_tab(state_manager, services):
    """
    Display the feedback history tab content
    
    Args:
        state_manager: State manager instance
        services (dict): Dictionary of services
    """
    import pandas as pd
    from ui.common import display_section_header, display_subsection_header
    
    display_section_header("Feedback History")
    
    # Get logger from services
    logger = services.get('logger')
    
    # Check if we have feedback history
    if logger and hasattr(logger, 'get_all_feedback'):
        # Get all feedback with detailed information
        feedback_data = logger.get_all_feedback()
        
        if feedback_data:
            # Create a DataFrame for better display
            feedback_df = pd.DataFrame(feedback_data)
            
            # Display feedback history
            st.dataframe(
                feedback_df,
                height=400,
                use_container_width=True,
                column_config={
                    'ID': st.column_config.NumberColumn(width="small"),
                    'Time': st.column_config.TextColumn(width="medium"),
                    'Type': st.column_config.TextColumn(width="medium"),
                    'Role': st.column_config.TextColumn(width="medium"),
                    'Job Description': st.column_config.TextColumn(width="medium"),
                    'Feedback': st.column_config.TextColumn(width="large")
                }
            )
            
            # Option to export feedback history
            if st.button("Export Feedback History", key="export_feedback_btn"):
                export_data = feedback_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=export_data,
                    file_name="feedback_history.csv",
                    mime="text/csv",
                    key="download_feedback_history"
                )
        else:
            st.info("No feedback history available yet.")
    else:
        # Show placeholder for missing logger
        st.warning("Feedback history tracking is not available.")
        
        # Display dummy data for preview
        dummy_data = {
            'ID': [1, 2, 3],
            'Time': ['2023-04-01 10:15:22', '2023-04-02 14:30:45', '2023-04-03 09:20:10'],
            'Type': ['Client Feedback', 'Hiring Manager Feedback', 'General Feedback'],
            'Role': ['Recruiter', 'Hiring Manager', 'HR Manager'],
            'Job Description': ['SoftwareDeveloper.txt', 'SoftwareDeveloper.txt', 'DataEngineer.txt'],
            'Feedback': [
                'Please add more emphasis on cloud technologies.',
                'We need to specify 5+ years of experience in the requirements.',
                'Update the job title to Senior Data Engineer for more visibility.'
            ]
        }
        
        st.dataframe(
            pd.DataFrame(dummy_data),
            height=400,
            use_container_width=True
        )