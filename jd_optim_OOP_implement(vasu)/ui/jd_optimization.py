import os
import streamlit as st
import datetime
from utils.file_utils import read_job_description, process_uploaded_docx, save_enhanced_jd
from utils.visualization import create_multi_radar_chart, create_comparison_dataframe
from ui.common import (
    display_section_header, display_subsection_header, 
    display_warning_message, display_info_message, display_success_message,
    switch_page
)

def render_jd_optimization_page(logger, analyzer, agent):
    """
    Render the unified JD Optimization page that consolidates the 
    job description enhancement and feedback workflows
    """
    display_section_header("📝 JD Optimization")
    
    ##########################
    # Part 1: Job Description Selection/Upload
    ##########################
    display_subsection_header("1. Select/Upload Job Description")
    
    # Create columns for file selection and file preview
    jd_directory = os.path.join(os.getcwd(), "JDs")
    try:
        files = [f for f in os.listdir(jd_directory) if f.endswith(('.txt', '.docx'))]
        file_col, upload_col = st.columns([2, 1])
        
        with file_col:
            selected_file = st.selectbox(
                "Select Job Description File", 
                files, 
                key="file_selector"
            )
        
        with upload_col:
            uploaded_file = st.file_uploader(
                "Or Upload New File", 
                type=['txt', 'docx'], 
                key="file_uploader"
            )
        
        # Handle uploaded file or selected file
        if uploaded_file:
            # Process uploaded file
            if uploaded_file.name.endswith('.txt'):
                content = uploaded_file.getvalue().decode('utf-8')
            else:  # .docx
                content = process_uploaded_docx(uploaded_file)
            
            st.session_state.original_jd = content
            selected_file = uploaded_file.name
            
            # Log file selection
            if logger.current_state["selected_file"] != selected_file:
                logger.log_file_selection(selected_file, content)
        elif selected_file:
            # Load selected file
            file_path = os.path.join(jd_directory, selected_file)
            
            # Reset state when file changes
            if st.session_state.get('last_file') != selected_file:
                st.session_state.last_file = selected_file
                if 'enhanced_versions' in st.session_state:
                    del st.session_state.enhanced_versions
                if 'original_jd' in st.session_state:
                    del st.session_state.original_jd
                st.session_state.reload_flag = True
            
            # Read the job description
            try:
                st.session_state.original_jd = read_job_description(file_path)
                
                # Log file selection
                if logger.current_state["selected_file"] != selected_file:
                    logger.log_file_selection(selected_file, st.session_state.original_jd)
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                return
    except FileNotFoundError:
        # If directory not found, allow direct file upload
        display_warning_message("Directory 'JDs' not found. You can upload a job description file directly.")
        uploaded_file = st.file_uploader(
            "Upload Job Description File", 
            type=['txt', 'docx'],
            key="file_uploader_alt"
        )
        
        if uploaded_file:
            # Process uploaded file
            if uploaded_file.name.endswith('.txt'):
                content = uploaded_file.getvalue().decode('utf-8')
            else:  # .docx
                content = process_uploaded_docx(uploaded_file)
            
            st.session_state.original_jd = content
            selected_file = uploaded_file.name
            
            # Log file selection
            if logger.current_state["selected_file"] != selected_file:
                logger.log_file_selection(selected_file, content)
        else:
            st.error("Please either upload a file or create a 'JDs' folder in the application directory.")
            return
    
    # Display original JD preview
    if 'original_jd' in st.session_state:
        st.markdown("<div class='subsection-header'>Job Description Preview</div>", unsafe_allow_html=True)
        
        # Show the JD preview in a collapsible section
        with st.expander("View Original Job Description", expanded=True):
            st.text_area(
                "Original Content", 
                st.session_state.original_jd, 
                height=250, 
                disabled=True,
                key="original_jd_display"
            )
    else:
        st.warning("Please select or upload a job description file.")
        return
    
    ##########################
    # Part 2: Generate Enhanced Versions
    ##########################
    display_subsection_header("2. Generate Enhanced Versions")
    
    generate_btn = st.button(
        "Generate Enhanced Versions", 
        type="primary", 
        key="generate_btn",
        help="Generate three AI-enhanced versions of your job description"
    )
    
    # Handle generating enhanced versions
    if generate_btn or ('enhanced_versions' not in st.session_state and st.session_state.get('reload_flag', False)):
        st.session_state.reload_flag = False
        with st.spinner("Generating enhanced versions... This may take a moment"):
            versions = agent.generate_initial_descriptions(st.session_state.original_jd)
            
            # Ensure we have 3 versions
            while len(versions) < 3:
                versions.append(f"Enhanced Version {len(versions)+1}:\n{st.session_state.original_jd}")
            
            st.session_state.enhanced_versions = versions
            logger.log_versions_generated(versions)
            st.rerun()
    
    # Display enhanced versions if available
    if 'enhanced_versions' in st.session_state and len(st.session_state.enhanced_versions) >= 3:
        # Create tabs for content and analysis
        enhanced_tabs = st.tabs(["Enhanced Versions", "Analysis & Comparison"])
        
        # Show enhanced versions tab content
        with enhanced_tabs[0]:
            version_tabs = st.tabs(["Version 1", "Version 2", "Version 3"])
            for idx, (tab, version) in enumerate(zip(version_tabs, st.session_state.enhanced_versions)):
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
            original_scores = analyzer.analyze_text(st.session_state.original_jd)
            intermediate_scores = {
                f'Version {i+1}': analyzer.analyze_text(version)
                for i, version in enumerate(st.session_state.enhanced_versions)
            }
            
            # Combine all scores for comparison
            all_scores = {'Original': original_scores, **intermediate_scores}
            
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
    
    ##########################
    # Part 3: Provide Feedback & Generate Final Version
    ##########################
    if 'enhanced_versions' in st.session_state and len(st.session_state.enhanced_versions) >= 3:
        display_subsection_header("3. Provide Feedback & Generate Final Version")
        
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
            
            # Display previous feedback if available
            if logger.current_state["feedback_history"]:
                st.markdown("**Previous Feedback:**")
                with st.expander("View Feedback History", expanded=False):
                    for i, feedback in enumerate(logger.current_state["feedback_history"], 1):
                        feedback_text = feedback.get("feedback", "") if isinstance(feedback, dict) else feedback
                        feedback_type = feedback.get("type", "General Feedback") if isinstance(feedback, dict) else "General Feedback"
                        st.markdown(f"**#{i} - {feedback_type}:**")
                        st.markdown(f"> {feedback_text}")
                        st.markdown("---")
        
        with right_col:
            # Feedback input
            # Define feedback types
            feedback_types = [
                "General Feedback", 
                "Rejected Candidate", 
                "Hiring Manager Feedback", 
                "Client Feedback", 
                "Selected Candidate", 
                "Interview Feedback"
            ]
            
            selected_feedback_type = st.selectbox(
                "Feedback Type:",
                options=feedback_types,
                index=feedback_types.index(st.session_state.feedback_type) if st.session_state.feedback_type in feedback_types else 0,
                key="feedback_type_selector"
            )
            
            # Update feedback type in session state
            if selected_feedback_type != st.session_state.feedback_type:
                st.session_state.feedback_type = selected_feedback_type
            
            # Manual feedback input
            user_feedback = st.text_area(
                "Enter your suggestions for improving the selected version:",
                height=150,
                placeholder="E.g., 'Add more emphasis on leadership skills', 'Include cloud technologies', etc.",
                key="user_feedback",
                help="Be specific about what you'd like to change or improve"
            )
            
            # Add feedback button
            if st.button("➕ Add Feedback", type="secondary", key="add_feedback"):
                if user_feedback.strip():
                    # Create feedback object
                    feedback_obj = {
                        "feedback": user_feedback,
                        "type": selected_feedback_type,
                        "role": st.session_state.role
                    }
                    
                    # Add to logger
                    logger.current_state["feedback_history"].append(feedback_obj)
                    logger._save_state()
                    
                    # Also track in session state for UI updates
                    if 'feedback_history' not in st.session_state:
                        st.session_state.feedback_history = []
                    st.session_state.feedback_history.append(feedback_obj)
                    
                    # Clear feedback input
                    st.session_state.clear_feedback = True
                    display_success_message("Feedback added successfully!")
                    st.rerun()
                else:
                    display_warning_message("Please enter some feedback first.")
        
        # Generate Final JD Button
        if st.button("🚀 Generate Final Enhanced Version", type="primary", key="generate_final_jd"):
            try:
                with st.spinner("Enhancing job description with feedback..."):
                    # Log version selection
                    logger.log_version_selection(selected_index)
                    
                    # Use the current enhanced version if it exists, otherwise use selected version
                    base_description = st.session_state.get("current_enhanced_version", 
                                                        st.session_state.enhanced_versions[selected_index])
                    
                    # Generate final JD using AI agent
                    final_description = agent.generate_final_description(
                        base_description, logger.current_state["feedback_history"]
                    )
                    
                    # Store new enhanced version in session state
                    st.session_state.current_enhanced_version = final_description
                    st.session_state.final_version_generated = True
                    st.session_state.final_version = final_description
                    
                    # Log the enhanced version
                    logger.log_enhanced_version(final_description, is_final=True)
                    display_success_message("Final version generated successfully!")
                    st.rerun()
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                return
        
        # Display Final Version if Generated
        if st.session_state.get('final_version_generated', False) and st.session_state.get('final_version'):
            final_description = st.session_state.final_version
            
            st.markdown("---")
            display_subsection_header("✅ Final Enhanced Job Description")
            
            # Display the final enhanced version
            st.text_area(
                "Final Content", 
                final_description, 
                height=400, 
                key="final_description"
            )
            
            # Compare original vs final JD with skill analysis
            display_subsection_header("📊 Final Analysis")
            
            # Calculate scores
            original_scores = analyzer.analyze_text(st.session_state.original_jd)
            final_scores = analyzer.analyze_text(final_description)
            
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
            
            # Download Final JD
            display_subsection_header("📥 Download Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="Download as TXT", 
                    data=final_description, 
                    file_name=f"enhanced_jd_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", 
                    mime="text/plain", 
                    key="download_txt"
                )
                logger.log_download("txt", f"enhanced_jd_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            
            with col2:
                if st.button("Download as DOCX", key="download_docx"):
                    docx_filename = f"enhanced_jd_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
                    save_enhanced_jd(final_description, docx_filename, 'docx')
                    display_success_message(f"Saved as {docx_filename}")
                    logger.log_download("docx", docx_filename)