import os
import streamlit as st
import datetime
import tempfile
import json
from docx import Document
from ui.common import (
    display_section_header, display_subsection_header,
    display_warning_message, display_info_message, display_success_message
)
from utils.file_utils import read_job_description, save_enhanced_jd
from utils.visualization import create_multi_radar_chart, create_comparison_dataframe

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

def process_uploaded_docx(uploaded_file):
    """Process an uploaded docx file and return its content"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(uploaded_file.getvalue())
        temp_path = tmp.name
    
    try:
        doc = Document(temp_path)
        content = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        return content
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def process_csv_content(csv_content, column_type="feedback"):
    """Extract content from CSV files based on column type"""
    try:
        import pandas as pd
        df = pd.read_csv(pd.StringIO(csv_content))
        
        # Define potential column names based on content type
        if column_type == "feedback":
            potential_columns = ['feedback', 'comments', 'notes', 'review', 'suggestions', 'input']
        else:  # job description
            potential_columns = ['job_description', 'description', 'jd', 'content', 'text']
        
        # Find the first matching column or use the first text column
        target_column = None
        for col in potential_columns:
            if col in df.columns:
                target_column = col
                break
        
        if target_column is None:
            # If no matching column found, use the first column that seems to have text content
            for col in df.columns:
                if df[col].dtype == 'object' and df[col].str.len().mean() > (50 if column_type == "jd" else 20):
                    target_column = col
                    break
        
        if target_column:
            if column_type == "feedback":
                # Combine all feedback entries
                combined_content = "\n\n".join(df[target_column].dropna().tolist())
                return combined_content, f"Extracted {len(df[target_column].dropna())} entries from column: {target_column}"
            else:
                # For JD, just use the first non-empty value
                return df[target_column].dropna().iloc[0], f"Extracted job description from column: {target_column}"
        
        # Fallback - concatenate all text columns
        text_cols = [col for col in df.columns if df[col].dtype == 'object']
        combined_content = "\n\n".join([f"{col}:\n{df[col].iloc[0]}" for col in text_cols[:5]])
        return combined_content, f"Could not identify a specific {column_type} column. Using combined text from CSV."
    except Exception as e:
        return csv_content, f"Error processing CSV structure: {str(e)}. Using raw CSV content."

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
    if 'client_enhanced_jd' not in st.session_state:
        st.session_state.client_enhanced_jd = None
    
    # Create two columns for file upload
    jd_col, feedback_col = st.columns(2)
    
    # --- Job Description Column ---
    with jd_col:
        display_subsection_header("Upload Job Description")
        
        # Option to use final enhanced version from the previous steps
        use_final_version = False
        if st.session_state.get('final_version'):
            use_final_version = st.checkbox(
                "Use previously generated final version", 
                value=False,
                help="Check this to use the final version you generated in the JD Optimization tab"
            )
        
        if use_final_version:
            jd_content = st.session_state.final_version
            st.success("Using previously generated final version.")
            st.session_state.client_jd = jd_content
            
            # Preview
            with st.expander("Preview Job Description", expanded=False):
                st.text_area("Job Description Content", jd_content, height=200, disabled=True)
        else:
            jd_file = st.file_uploader(
                "📄 Drop or upload a Job Description",
                type=["txt", "docx", "csv"],
                key="client_jd_upload",
                help="Upload the job description you want to enhance (TXT, DOCX, or CSV)"
            )
            
            # Process uploaded JD file if present
            if jd_file:
                try:
                    if jd_file.name.endswith(".txt"):
                        job_description = jd_file.getvalue().decode("utf-8")
                    elif jd_file.name.endswith(".csv"):
                        # Special handling for CSV
                        csv_content = jd_file.getvalue().decode("utf-8")
                        job_description, message = process_csv_content(csv_content, "jd")
                        st.info(message)
                    elif jd_file.name.endswith(".docx"):
                        job_description = process_uploaded_docx(jd_file)
                    else:
                        st.error("Unsupported file format.")
                        return
                    
                    # Store in session state
                    st.session_state.client_jd = job_description
                    
                    # Preview
                    with st.expander("Preview Job Description", expanded=False):
                        st.text_area("Job Description Content", job_description, height=200, disabled=True)
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
    
    # --- Client Feedback Column ---
    with feedback_col:
        display_subsection_header("Upload Client Feedback")
        
        # Create tabs for feedback methods
        feedback_tabs = st.tabs(["Upload Feedback File", "Select from Directory", "Enter Manually"])
        
        # Define feedback types
        feedback_types = [
            "Client Feedback", 
            "Rejected Candidate Feedback", 
            "Hiring Manager Feedback", 
            "Selected Candidate Feedback", 
            "Interview Feedback"
        ]
        
        # 1. Upload feedback file tab
        with feedback_tabs[0]:
            feedback_file = st.file_uploader(
                "📝 Drop or upload Feedback File",
                type=["txt", "docx", "csv"],
                key="client_feedback_upload",
                help="Upload the feedback from your client (TXT, DOCX, or CSV)"
            )
            
            # Feedback type selection
            selected_feedback_type = st.selectbox(
                "Feedback Type:",
                options=feedback_types,
                index=feedback_types.index(st.session_state.client_feedback_type_value) 
                    if st.session_state.client_feedback_type_value in feedback_types else 0,
                key="client_feedback_type"
            )
            
            # Update session state
            st.session_state.client_feedback_type_value = selected_feedback_type
            
            # Process uploaded feedback file
            if feedback_file:
                try:
                    if feedback_file.name.endswith(".txt"):
                        client_feedback = feedback_file.getvalue().decode("utf-8")
                    elif feedback_file.name.endswith(".csv"):
                        # Special handling for CSV
                        csv_content = feedback_file.getvalue().decode("utf-8")
                        client_feedback, message = process_csv_content(csv_content, "feedback")
                        st.info(message)
                    elif feedback_file.name.endswith(".docx"):
                        client_feedback = process_uploaded_docx(feedback_file)
                    else:
                        st.error("Unsupported file format.")
                        return
                    
                    # Store in session state
                    st.session_state.client_feedback = client_feedback
                    
                    # Preview
                    with st.expander("Preview Feedback", expanded=True):
                        st.text_area("Feedback Content", client_feedback, height=200, disabled=True)
                except Exception as e:
                    st.error(f"Error reading feedback file: {str(e)}")
        
        # 2. Select from directory tab
        with feedback_tabs[1]:
            feedback_directory = os.path.join(os.getcwd(), "Feedbacks")
            
            if not os.path.exists(feedback_directory):
                display_warning_message("The 'Feedbacks' directory does not exist. Create it or upload a file directly.")
            else:
                try:
                    feedback_files = [f for f in os.listdir(feedback_directory) 
                                    if f.endswith(('.txt', '.docx', '.csv'))]
                    
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
                        
                        # Select feedback type
                        file_feedback_type = st.selectbox(
                            "Feedback Type:",
                            options=feedback_types,
                            index=feedback_types.index(st.session_state.client_feedback_type_value)
                                if st.session_state.client_feedback_type_value in feedback_types else 0,
                            key="file_feedback_type"
                        )
                        
                        # Add a button to load the selected file
                        if st.button("Load Selected File", key="load_feedback_file"):
                            if selected_feedback_file:
                                feedback_path = os.path.join(feedback_directory, selected_feedback_file)
                                
                                if not os.path.exists(feedback_path):
                                    st.error(f"File not found: {feedback_path}")
                                else:
                                    try:
                                        if selected_feedback_file.endswith('.csv'):
                                            with open(feedback_path, 'r', encoding='utf-8') as f:
                                                csv_content = f.read()
                                            feedback_content, message = process_csv_content(csv_content, "feedback")
                                            st.info(message)
                                        else:
                                            feedback_content = read_feedback_file(feedback_path)
                                        
                                        # Store in session state
                                        st.session_state.client_feedback = feedback_content
                                        st.session_state.client_feedback_type_value = file_feedback_type
                                        
                                        # Display preview
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
                except Exception as e:
                    st.error(f"Error accessing Feedbacks directory: {str(e)}")
        
        # 3. Manual input tab
        with feedback_tabs[2]:
            manual_feedback = st.text_area(
                "Enter client feedback:",
                height=200,
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
    
    # --- Generate Enhanced JD Button ---
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
        if not st.session_state.client_jd:
            st.warning("Please provide a job description before generating.")
            return
            
        if not st.session_state.client_feedback:
            st.warning("Please provide client feedback before generating.")
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
                
                # Create the prompt for the agent
                prompt = (
                    "You are an expert in job description refinement.\n\n"
                    "Please revise the provided job description **only based on the feedback** given by the client.\n\n"
                    "Do not introduce any information or changes not explicitly stated in the feedback.\n"
                    "Only make edits that directly reflect specific feedback content.\n\n"
                    "**Guidelines:**\n"
                    "- Do not make assumptions.\n"
                    "- Do not change formatting or structure unless required by feedback.\n"
                    "- Refer to the position as 'this role'.\n"
                    "- If the feedback is vague or irrelevant, leave the job description unchanged.\n\n"
                    f"### Original Job Description:\n{job_description}\n\n"
                    f"### Client Feedback:\n{client_feedback}\n\n"
                    "### Please return only the revised job description below (leave unchanged if no edits are needed):\n"
                )
                
                # Call the agent to generate the enhanced JD
                native_request = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1024,
                    "temperature": 0.7,
                    "messages": [{"role": "user", "content": prompt}],
                }
                
                response = agent.client.invoke_model(
                    modelId=agent.model_id,
                    body=json.dumps(native_request),
                    contentType="application/json",
                )
                response_body = json.loads(response["body"].read().decode("utf-8"))
                
                if isinstance(response_body, dict) and "content" in response_body:
                    content = response_body["content"]
                    if isinstance(content, list):
                        enhanced_jd = " ".join([item.get("text", "") for item in content]).strip()
                    else:
                        enhanced_jd = content if isinstance(content, str) else "[No valid content returned]"
                else:
                    enhanced_jd = "[Unexpected response format]"
                
                # Store the enhanced JD in session state
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
    
    # --- Display Results ---
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