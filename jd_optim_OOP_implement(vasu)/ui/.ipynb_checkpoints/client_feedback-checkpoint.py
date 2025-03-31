import os
import streamlit as st
import datetime
from docx import Document
from ui.common import (
    display_section_header, display_subsection_header,
    display_warning_message, display_info_message, display_success_message
)
from utils.file_utils import read_job_description, save_enhanced_jd

def read_feedback_file(file_path):
    """Read feedback from a file"""
    if file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    elif file_path.endswith('.docx'):
        doc = Document(file_path)
        return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    elif file_path.endswith('.csv'):
        # Read CSV as raw text first
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    else:
        raise ValueError("Unsupported file format")

def render_client_feedback_page(logger, analyzer, agent):
    """Render the Client Feedback tab with JD + Feedback drop zones"""
    display_section_header("💬 Client Feedback Enhancement")
    
    # Initialize session state variables if they don't exist
    if 'client_jd' not in st.session_state:
        st.session_state.client_jd = ""
    if 'client_feedback' not in st.session_state:
        st.session_state.client_feedback = ""
    if 'client_feedback_type_value' not in st.session_state:
        st.session_state.client_feedback_type_value = "Client Feedback"
    
    # Create two columns for file upload
    jd_col, feedback_col = st.columns(2)
    
    with jd_col:
        display_subsection_header("Upload Job Description")
        jd_file = st.file_uploader(
            "📄 Drop or upload a Job Description",
            type=["txt", "docx", "csv"],
            key="client_jd_upload",
            help="Upload the job description you want to enhance (TXT, DOCX, or CSV)"
        )
        
        # Optional: Display JD preview if uploaded
        if jd_file:
            try:
                if jd_file.name.endswith(".txt"):
                    job_description = jd_file.getvalue().decode("utf-8")
                elif jd_file.name.endswith(".csv"):
                    # Handle CSV file as a special case
                    try:
                        import pandas as pd
                        # Read the CSV file
                        csv_content = jd_file.getvalue().decode("utf-8")
                        
                        # Let user know we're processing a CSV
                        st.info("Processing CSV file. Attempting to extract job description content...")
                        
                        # Try to intelligently extract a job description from CSV
                        # This is a simplified approach - in a real app you'd want more robust handling
                        df = pd.read_csv(pd.StringIO(csv_content))
                        
                        # Look for columns that might contain job description content
                        potential_columns = ['job_description', 'description', 'jd', 'content', 'text']
                        
                        # Find the first matching column or use the first text column
                        jd_column = None
                        for col in potential_columns:
                            if col in df.columns:
                                jd_column = col
                                break
                        
                        if jd_column is None:
                            # If no matching column found, use the first column that seems to have text content
                            for col in df.columns:
                                if df[col].dtype == 'object' and df[col].str.len().mean() > 50:
                                    jd_column = col
                                    break
                        
                        if jd_column:
                            # Use the first non-empty value in the column
                            job_description = df[jd_column].dropna().iloc[0]
                            st.success(f"Extracted job description from column: {jd_column}")
                        else:
                            # Fallback - concatenate all text columns
                            text_cols = [col for col in df.columns if df[col].dtype == 'object']
                            job_description = "\n\n".join([f"{col}:\n{df[col].iloc[0]}" for col in text_cols[:5]])
                            st.warning("Could not identify a specific job description column. Using combined text from CSV.")
                    except Exception as e:
                        st.error(f"Error processing CSV file: {str(e)}")
                        job_description = csv_content  # Use raw CSV content as fallback
                else:  # .docx
                    # Save to temporary file to use python-docx
                    temp_path = f"temp_{jd_file.name}"
                    with open(temp_path, 'wb') as f:
                        f.write(jd_file.getvalue())
                    
                    doc = Document(temp_path)
                    job_description = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
                    
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                
                # Store in session state for later use
                st.session_state.client_jd = job_description
                
                # Preview
                with st.expander("Preview Job Description", expanded=False):
                    st.text_area("Job Description Content", job_description, height=200, disabled=True)
            except Exception as e:
                st.error(f"Error reading job description file: {str(e)}")
    
    with feedback_col:
        display_subsection_header("Upload Client Feedback")
        
        # Create tabs for selecting from directory or uploading
        feedback_tabs = st.tabs(["Upload Feedback File", "Select from Feedbacks Folder"])
        
        # Define feedback types
        feedback_types = [
            "Client Feedback", 
            "Rejected Candidate Feedback", 
            "Hiring Manager Feedback", 
            "Selected Candidate Feedback", 
            "Interview Feedback"
        ]
        
        with feedback_tabs[0]:
            # Upload file option
            feedback_file = st.file_uploader(
                "📝 Drop or upload Client Feedback",
                type=["txt", "docx", "csv"],
                key="client_feedback_upload",
                help="Upload the feedback received from your client (TXT, DOCX, or CSV)"
            )
            
            # Feedback type selection - use a consistent variable name across tabs
            selected_feedback_type = st.selectbox(
                "Feedback Type:",
                options=feedback_types,
                index=feedback_types.index(st.session_state.client_feedback_type_value) 
                    if st.session_state.client_feedback_type_value in feedback_types else 0,
                key="client_feedback_type"
            )
            
            # Update the session state with the selected value
            st.session_state.client_feedback_type_value = selected_feedback_type
            
            # Optional: Display feedback preview if uploaded
            if feedback_file:
                try:
                    if feedback_file.name.endswith(".txt"):
                        client_feedback = feedback_file.getvalue().decode("utf-8")
                    elif feedback_file.name.endswith(".csv"):
                        # Handle CSV file as a special case
                        try:
                            import pandas as pd
                            # Read the CSV file
                            csv_content = feedback_file.getvalue().decode("utf-8")
                            
                            # Let user know we're processing a CSV
                            st.info("Processing CSV file. Attempting to extract feedback content...")
                            
                            # Try to intelligently extract feedback from CSV
                            df = pd.read_csv(pd.StringIO(csv_content))
                            
                            # Look for columns that might contain feedback content
                            potential_columns = ['feedback', 'comments', 'notes', 'review', 'suggestions', 'input']
                            
                            # Find the first matching column or use the first text column
                            feedback_column = None
                            for col in potential_columns:
                                if col in df.columns:
                                    feedback_column = col
                                    break
                            
                            if feedback_column is None:
                                # If no matching column found, use the first column that seems to have text content
                                for col in df.columns:
                                    if df[col].dtype == 'object' and df[col].str.len().mean() > 20:
                                        feedback_column = col
                                        break
                            
                            if feedback_column:
                                # Combine all feedback entries
                                combined_feedback = "\n\n".join(df[feedback_column].dropna().tolist())
                                client_feedback = combined_feedback
                                st.success(f"Extracted {len(df[feedback_column].dropna())} feedback entries from column: {feedback_column}")
                            else:
                                # Fallback - concatenate all text columns
                                text_cols = [col for col in df.columns if df[col].dtype == 'object']
                                client_feedback = "\n\n".join([f"{col}:\n{df[col].iloc[0]}" for col in text_cols[:5]])
                                st.warning("Could not identify a specific feedback column. Using combined text from CSV.")
                        except Exception as e:
                            st.error(f"Error processing CSV file: {str(e)}")
                            client_feedback = csv_content  # Use raw CSV content as fallback
                    else:  # .docx
                        # Save to temporary file to use python-docx
                        temp_path = f"temp_{feedback_file.name}"
                        with open(temp_path, 'wb') as f:
                            f.write(feedback_file.getvalue())
                        
                        doc = Document(temp_path)
                        client_feedback = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
                        
                        # Clean up temp file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                    
                    # Store in session state for later use
                    st.session_state.client_feedback = client_feedback
                    
                    # Preview
                    with st.expander("Preview Client Feedback", expanded=True):
                        st.text_area("Feedback Content", client_feedback, height=200, disabled=True)
                except Exception as e:
                    st.error(f"Error reading feedback file: {str(e)}")
        
        with feedback_tabs[1]:
            # Select from Feedbacks folder
            feedback_directory = os.path.join(os.getcwd(), "Feedbacks")
            
            # Check if directory exists
            if not os.path.exists(feedback_directory):
                display_warning_message("The 'Feedbacks' directory does not exist. Please create it or upload a file directly.")
            else:
                # Get all .txt, .docx, and .csv files from the Feedbacks folder
                try:
                    feedback_files = [f for f in os.listdir(feedback_directory) 
                                    if f.endswith(('.txt', '.docx', '.csv'))]
                except Exception as e:
                    st.error(f"Error accessing Feedbacks directory: {str(e)}")
                    feedback_files = []
                
                if not feedback_files:
                    display_warning_message("No feedback files found in the Feedbacks directory.")
                else:
                    # Allow user to select a feedback file
                    selected_feedback_file = st.selectbox(
                        "Select Feedback File",
                        feedback_files,
                        help="Choose a feedback file to process",
                        key="folder_feedback_file"
                    )
                    
                    # Select feedback type for file - use a different key
                    file_feedback_type = st.selectbox(
                        "File Feedback Type:",
                        options=feedback_types,
                        index=feedback_types.index(st.session_state.client_feedback_type_value)
                            if st.session_state.client_feedback_type_value in feedback_types else 0,
                        key="file_feedback_type"
                    )
                    
                    # Add a button to load the selected file
                    if st.button("Load Selected File", key="load_feedback_file"):
                        if selected_feedback_file:
                            feedback_path = os.path.join(feedback_directory, selected_feedback_file)
                            
                            # Check if file exists
                            if not os.path.exists(feedback_path):
                                st.error(f"File not found: {feedback_path}")
                            else:
                                # Extract text based on file type
                                try:
                                    # Use simple read function first
                                    feedback_content = read_feedback_file(feedback_path)
                                    
                                    # Process CSV files specially
                                    if selected_feedback_file.endswith('.csv'):
                                        try:
                                            import pandas as pd
                                            df = pd.read_csv(feedback_path)
                                            
                                            # Look for columns that might contain feedback content
                                            potential_columns = ['feedback', 'comments', 'notes', 'review', 'suggestions', 'input']
                                            
                                            # Find the first matching column or use the first text column
                                            feedback_column = None
                                            for col in potential_columns:
                                                if col in df.columns:
                                                    feedback_column = col
                                                    break
                                            
                                            if feedback_column is None:
                                                # If no matching column found, use the first column that seems to have text content
                                                for col in df.columns:
                                                    if df[col].dtype == 'object' and df[col].str.len().mean() > 20:
                                                        feedback_column = col
                                                        break
                                            
                                            if feedback_column:
                                                # Combine all feedback entries
                                                processed_feedback = "\n\n".join(df[feedback_column].dropna().tolist())
                                                st.success(f"Extracted {len(df[feedback_column].dropna())} feedback entries from column: {feedback_column}")
                                            else:
                                                # Fallback - concatenate all text columns
                                                text_cols = [col for col in df.columns if df[col].dtype == 'object']
                                                processed_feedback = "\n\n".join([f"{col}:\n{df[col].iloc[0]}" for col in text_cols[:5]])
                                                st.warning("Could not identify a specific feedback column. Using combined text from CSV.")
                                                
                                            # Update feedback content with processed version
                                            feedback_content = processed_feedback
                                        except Exception as e:
                                            st.warning(f"Couldn't process CSV structure: {str(e)}. Using raw CSV content instead.")
                                    
                                    # Store in session state
                                    st.session_state.client_feedback = feedback_content
                                    st.session_state.client_feedback_type_value = file_feedback_type
                                    
                                    # Display the feedback content
                                    st.text_area(
                                        "Feedback Content",
                                        feedback_content,
                                        height=200,
                                        disabled=True,
                                        key="folder_feedback_content"
                                    )
                                    
                                    st.success(f"Successfully loaded feedback from {selected_feedback_file}")
                                except Exception as e:
                                    st.error(f"Error reading feedback file: {str(e)}")
        
        # Allow manual feedback input as an alternative to file upload
        with st.expander("Or Enter Feedback Manually", expanded=False):
            manual_feedback = st.text_area(
                "Enter client feedback:",
                height=150,
                placeholder="Enter the feedback from your client here...",
                key="manual_client_feedback"
            )
            
            manual_feedback_type = st.selectbox(
                "Feedback Type:",
                options=feedback_types,
                index=feedback_types.index(st.session_state.client_feedback_type_value)
                    if st.session_state.client_feedback_type_value in feedback_types else 0,
                key="manual_feedback_type"
            )
            
            if st.button("Use This Feedback", key="use_manual_feedback"):
                if manual_feedback.strip():
                    st.session_state.client_feedback = manual_feedback
                    st.session_state.client_feedback_type_value = manual_feedback_type
                    st.success("Manual feedback saved!")
                else:
                    st.warning("Please enter some feedback first.")

    # Generate enhanced JD button
    st.markdown("---")
    display_subsection_header("Generate Enhanced Job Description")
    
    generate_col1, generate_col2 = st.columns([3, 1])
    
    with generate_col1:
        generate_btn = st.button(
            "🚀 Generate Enhanced Job Description", 
            type="primary", 
            key="generate_client_enhanced_jd",
            help="Generate an enhanced version of the job description based on client feedback"
        )
        
    with generate_col2:
        st.caption("AI will enhance the job description based on the provided client feedback.")
    
    # Handle generation process
    if generate_btn:
        if not st.session_state.client_jd or not st.session_state.client_feedback:
            st.warning("Please upload both a job description and client feedback before generating.")
            return
        
        job_description = st.session_state.client_jd
        client_feedback = st.session_state.client_feedback
        feedback_type = st.session_state.client_feedback_type_value
        
        with st.spinner("Enhancing job description with client feedback..."):
            try:
                # Create feedback object with type
                feedback_obj = {
                    "feedback": client_feedback,
                    "type": feedback_type,
                    "role": st.session_state.role
                }
                
                # Add to the logger's feedback history
                logger.current_state["feedback_history"].append(feedback_obj)
                logger._save_state()
                
                # Log the action
                logger.current_state["actions"].append({
                    "action": "client_feedback",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "feedback_type": feedback_type
                })
                logger._save_state()
                
                # Generate enhanced JD using the agent
                enhanced_jd = agent.generate_final_description(
                    job_description,
                    [feedback_obj]
                )
                
                # Store the enhanced JD
                st.session_state.client_enhanced_jd = enhanced_jd
                
                # Log the enhanced version
                logger.log_enhanced_version(enhanced_jd, is_final=True)
                
                # Display success message
                display_success_message("Job description enhanced successfully based on client feedback!")
                
                # Force page refresh to show results
                st.rerun()
            except Exception as e:
                st.error(f"Error enhancing job description: {str(e)}")
                st.error("Please try again or contact support if the problem persists.")
    
    # Display results if generation was successful
    if 'client_enhanced_jd' in st.session_state and st.session_state.client_enhanced_jd:
        # Display results in an organized layout
        st.markdown("---")
        display_section_header("Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            display_subsection_header("Original Job Description")
            st.text_area(
                "Original Content",
                st.session_state.client_jd,
                height=300,
                disabled=True,
                key="client_original_jd_display"
            )
        
        with col2:
            display_subsection_header("Enhanced Job Description")
            st.text_area(
                "Enhanced Content",
                st.session_state.client_enhanced_jd,
                height=300,
                key="client_enhanced_jd_display"
            )
        
        # Compare original vs enhanced with skill analysis
        if st.session_state.client_jd and st.session_state.client_enhanced_jd:
            display_section_header("Comparison Analysis")
            
            # Analyze both versions
            original_scores = analyzer.analyze_text(st.session_state.client_jd)
            enhanced_scores = analyzer.analyze_text(st.session_state.client_enhanced_jd)
            
            from utils.visualization import create_multi_radar_chart, create_comparison_dataframe
            
            comp_col1, comp_col2 = st.columns([1, 1])
            
            with comp_col1:
                display_subsection_header("Skill Coverage Comparison")
                radar_chart = create_multi_radar_chart({'Original': original_scores, 'Enhanced': enhanced_scores})
                st.plotly_chart(radar_chart, use_container_width=True, key="client_radar")
            
            with comp_col2:
                display_subsection_header("Detailed Analysis")
                comparison_df = create_comparison_dataframe({'Original': original_scores, 'Enhanced': enhanced_scores})
                st.dataframe(
                    comparison_df,
                    height=400,
                    use_container_width=True,
                    hide_index=True,
                    key="client_comparison"
                )
                st.caption("Percentages indicate keyword coverage in each category")
        
        # Download options
        display_section_header("Download Options")
        
        download_col1, download_col2 = st.columns(2)
        
        with download_col1:
            st.download_button(
                label="Download as TXT",
                data=st.session_state.client_enhanced_jd,
                file_name=f"client_enhanced_jd_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                key="client_download_txt"
            )
            logger.log_download("txt", f"client_enhanced_jd_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
        with download_col2:
            if st.button("Download as DOCX", key="client_download_docx"):
                docx_filename = f"client_enhanced_jd_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
                save_enhanced_jd(st.session_state.client_enhanced_jd, docx_filename, 'docx')
                display_success_message(f"Saved as {docx_filename}")
                logger.log_download("docx", docx_filename)