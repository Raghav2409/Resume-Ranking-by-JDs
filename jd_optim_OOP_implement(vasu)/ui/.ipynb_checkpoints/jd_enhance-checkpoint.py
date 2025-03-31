import os
import streamlit as st
import datetime
from utils.file_utils import read_job_description, process_uploaded_docx, get_jd_files, save_enhanced_jd
from utils.visualization import create_multi_radar_chart, create_comparison_dataframe
from ui.common import display_section_header, display_warning_message, display_info_message, switch_page

def render_jd_enhance_page(logger, analyzer, agent):
    """Render the JD enhancement page"""
    display_section_header("📄 Job Description Selection")
    
    jd_directory = os.path.join(os.getcwd(), "JDs")
    try:
        files = get_jd_files()
        
        # Create columns for file selection and file preview
        file_col, upload_col = st.columns([2, 1])

        with file_col:
            selected_file = st.selectbox(
                "Select Job Description File",
                files,
                help="Choose a job description file to enhance",
                key="file_selector"
            )

        with upload_col:
            # Add option to upload a file directly
            uploaded_file = st.file_uploader(
                "Or Upload New File",
                type=['txt', 'docx'],
                help="Upload a new job description file"
            )
        
        # Handle file selection or upload
        if uploaded_file:
            if uploaded_file.name.endswith('.txt'):
                content = uploaded_file.getvalue().decode('utf-8')
            else:  # .docx
                content = process_uploaded_docx(uploaded_file)
            
            st.session_state.original_jd = content
            selected_file = uploaded_file.name
            
            # Log file selection (only if changed)
            if logger.current_state["selected_file"] != selected_file:
                logger.log_file_selection(selected_file, content)
        elif selected_file:
            file_path = os.path.join(jd_directory, selected_file)
            try:
                # Reset state when file changes
                if st.session_state.last_file != selected_file:
                    st.session_state.last_file = selected_file
                    st.session_state.pop('enhanced_versions', None)
                    st.session_state.pop('original_jd', None)
                    st.session_state.reload_flag = True
                    content = read_job_description(file_path)  # Store the result in a variable
                    st.session_state.original_jd = content
                    
                    # Log file selection (only if changed)
                    if logger.current_state["selected_file"] != selected_file:
                        logger.log_file_selection(selected_file, content)  # Now content is defined
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                return
    except FileNotFoundError:
        # If directory not found, allow direct file upload
        display_warning_message("Directory 'JDs' not found. You can upload a job description file directly.")
        uploaded_file = st.file_uploader("Upload Job Description File", type=['txt', 'docx'])
        
        if uploaded_file:
            if uploaded_file.name.endswith('.txt'):
                content = uploaded_file.getvalue().decode('utf-8')
            else:  # .docx
                content = process_uploaded_docx(uploaded_file)
            
            st.session_state.original_jd = content
            selected_file = uploaded_file.name
            
            # Log file selection (only if changed)
            if logger.current_state["selected_file"] != selected_file:
                logger.log_file_selection(selected_file, content)
        else:
            st.error("Please either upload a file or create a 'JDs' folder in the application directory.")
            return

    # From here, the rest of the app continues with either the uploaded or selected file
    if 'original_jd' in st.session_state:
        original_jd = st.session_state.original_jd
        
        # Display original JD with better formatting
        display_section_header("Original Job Description")
        
        # Create tabs for original content and quick preview
        original_tabs = st.tabs(["Full Content", "Quick Preview"])
        
        with original_tabs[0]:
            st.text_area(
                "Original Content", 
                original_jd, 
                height=250, 
                disabled=True,
                key="original_jd_display"
            )
            
        with original_tabs[1]:
            # Show a preview with just first 500 characters
            preview_length = min(500, len(original_jd))
            st.write(original_jd[:preview_length] + ("..." if len(original_jd) > preview_length else ""))
        
        # Generate enhanced versions
        display_section_header("✨ Enhanced Versions")
        
        generate_col1, generate_col2 = st.columns([3, 1])
        
        with generate_col1:
            generate_btn = st.button(
                "Generate Enhanced Versions", 
                type="primary", 
                key="generate_btn",
                help="Generate three AI-enhanced versions of your job description"
            )
            
        with generate_col2:
            st.caption("AI will create three distinct improved versions of your job description for you to review.")
        
        # Handle generating enhanced versions
        if generate_btn or ('enhanced_versions' not in st.session_state and st.session_state.reload_flag):
            st.session_state.reload_flag = False
            with st.spinner("Generating enhanced versions... This may take a moment"):
                versions = agent.generate_initial_descriptions(original_jd)
                
                # Ensure we have 3 versions
                while len(versions) < 3:
                    versions.append(f"Enhanced Version {len(versions)+1}:\n{original_jd}")
                
                st.session_state.enhanced_versions = versions
                logger.log_versions_generated(versions)
                st.rerun()

        # Display enhanced versions and their analysis
        if 'enhanced_versions' in st.session_state and len(st.session_state.enhanced_versions) >= 3:
            # Analyze all versions
            original_scores = analyzer.analyze_text(original_jd)
            intermediate_scores = {
                f'Version {i+1}': analyzer.analyze_text(version)
                for i, version in enumerate(st.session_state.enhanced_versions)
            }
            
            # Combine all scores for comparison
            all_scores = {'Original': original_scores, **intermediate_scores}

            # Create tabs for content and analysis
            enhanced_tabs = st.tabs(["Enhanced Versions", "Analysis & Comparison"])
                
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
        
            with enhanced_tabs[1]:
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
                    
