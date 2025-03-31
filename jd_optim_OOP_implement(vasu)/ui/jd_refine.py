import os
import json
import streamlit as st
import datetime
from docx import Document
from utils.file_utils import read_job_description, save_enhanced_jd
from utils.visualization import create_multi_radar_chart, create_comparison_dataframe
from ui.common import (
    display_section_header, display_subsection_header,
    display_warning_message, display_info_message, display_success_message,
    switch_page
)

def display_filtered_feedback_history():
    """Display feedback history with filtering options"""
    # Get all available sessions
    sessions = st.session_state.logger.list_sessions()
    
    if not sessions:
        display_info_message("No previous feedback found")
        return
    
    # Create a list to store all feedback data
    all_feedback = []
    
    # Collect unique values for filters
    unique_roles = set()
    unique_files = set()
    unique_dates = set()
    unique_feedback_types = set()
    
    # Loop through each session to collect feedback
    for session_info in sessions:
        session_id = session_info["session_id"]
        try:
            # Load the session data directly from file without creating a new logger instance
            log_file = os.path.join("logs", f"jdoptim_session_{session_id}.json")
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    state = json.load(f)
                
                role = state.get("username", "Unknown Role")
                unique_roles.add(role)
                
                file_name = state.get("selected_file", "Unknown")
                unique_files.add(file_name)
                
                session_date = state.get("session_start_time", "Unknown")
                # Extract just the date part if it's a full timestamp
                if isinstance(session_date, str) and "T" in session_date:
                    session_date = session_date.split("T")[0]
                unique_dates.add(session_date)
                
                # Add each feedback item with metadata
                for i, feedback in enumerate(state.get("feedback_history", [])):
                    # Get timestamp for the feedback if available
                    feedback_time = "Unknown"
                    for action in state.get("actions", []):
                        if action.get("action") == "feedback" and action.get("index", -1) == i:
                            feedback_time = action.get("timestamp", "Unknown")
                            break
                    
                    # Handle different feedback formats (string or dict)
                    if isinstance(feedback, dict):
                        feedback_content = feedback.get("feedback", "")
                        feedback_type = feedback.get("type", "General Feedback")
                        feedback_role = feedback.get("role", role)
                    else:
                        feedback_content = feedback
                        feedback_type = "General Feedback"
                        feedback_role = role
                    
                    # Add to unique feedback types
                    unique_feedback_types.add(feedback_type)
                    
                    all_feedback.append({
                        "Role": feedback_role,
                        "File": file_name,
                        "Session Date": session_date,
                        "Feedback Time": feedback_time,
                        "Feedback Type": feedback_type,
                        "Feedback": feedback_content
                    })
        except Exception as e:
            print(f"Error reading session {session_id}: {str(e)}")
    
    if not all_feedback:
        display_info_message("No feedback found in any session")
        return
            
    # Convert to DataFrame
    import pandas as pd
    feedback_df = pd.DataFrame(all_feedback)
    
    # Sort by most recent first if timestamps are available
    if "Feedback Time" in feedback_df.columns:
        try:
            # Parse timestamps where possible
            parsed_timestamps = []
            for timestamp in feedback_df["Feedback Time"]:
                try:
                    if isinstance(timestamp, str) and "T" in timestamp:
                        dt = datetime.datetime.fromisoformat(timestamp)
                        parsed_timestamps.append(dt)
                    else:
                        parsed_timestamps.append(datetime.datetime(1900, 1, 1))
                except:
                    parsed_timestamps.append(datetime.datetime(1900, 1, 1))
            
            feedback_df["Parsed Timestamp"] = parsed_timestamps
            feedback_df = feedback_df.sort_values("Parsed Timestamp", ascending=False)
        except:
            pass  # If sorting fails, just use the original order
    
    # Create filters with a container to keep UI clean
    with st.expander("Filter Feedback", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Role filter - make sure we handle None values in the unique_roles set
            cleaned_roles = [role for role in unique_roles if role is not None]
            selected_roles = st.multiselect(
                "Filter by Role:",
                options=sorted(cleaned_roles),
                default=[],
                key="filter_roles"
            )
        
        with col2:
            # File filter - make sure we handle None values in the unique_files set
            cleaned_files = [file for file in unique_files if file is not None]
            selected_files = st.multiselect(
                "Filter by Job Description:",
                options=sorted(cleaned_files),
                default=[],
                key="filter_files"
            )
        
        with col3:
            # Feedback type filter
            cleaned_feedback_types = [ft for ft in unique_feedback_types if ft is not None]
            selected_feedback_types = st.multiselect(
                "Filter by Feedback Type:",
                options=sorted(cleaned_feedback_types),
                default=[],
                key="filter_types"
            )
        
        # Text search
        search_text = st.text_input("Search in feedback:", "", key="search_feedback")
    
    # Apply filters
    filtered_df = feedback_df.copy()
    
    if selected_roles:
        filtered_df = filtered_df[filtered_df["Role"].isin(selected_roles)]
    
    if selected_files:
        filtered_df = filtered_df[filtered_df["File"].isin(selected_files)]
    
    if selected_feedback_types:
        filtered_df = filtered_df[filtered_df["Feedback Type"].isin(selected_feedback_types)]
    
    if search_text:
        filtered_df = filtered_df[filtered_df["Feedback"].str.contains(search_text, case=False, na=False)]
    
    # Show filter summary
    st.write(f"Showing {len(filtered_df)} of {len(feedback_df)} feedback items")
    
    # Display the filtered dataframe
    if not filtered_df.empty:
        # Format timestamps for display
        readable_timestamps = []
        for timestamp in filtered_df["Feedback Time"]:
            try:
                if isinstance(timestamp, str) and "T" in timestamp:
                    dt = datetime.datetime.fromisoformat(timestamp)
                    readable_timestamps.append(dt.strftime("%Y-%m-%d %H:%M:%S"))
                else:
                    readable_timestamps.append(str(timestamp))
            except:
                readable_timestamps.append(str(timestamp))
        
        filtered_df["Formatted Time"] = readable_timestamps
        
        # Display dataframe
        st.dataframe(
            filtered_df,
            use_container_width=True,
            column_config={
                "Role": st.column_config.TextColumn("Role"),
                "File": st.column_config.TextColumn("Job Description"),
                "Formatted Time": st.column_config.TextColumn("Time"),
                "Feedback Type": st.column_config.TextColumn("Feedback Type"),
                "Feedback": st.column_config.TextColumn("Feedback Content", width="large"),
            },
            hide_index=True
        )
    else:
        display_info_message("No feedback matches the selected filters")
    
    # Option to export filtered results
    if not filtered_df.empty:
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Export Filtered Feedback",
            data=csv,
            file_name=f"feedback_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def render_jd_refine_page(logger, analyzer, agent):
    """Render the JD refinement and feedback page"""
    
    # Add navigation breadcrumb with return option
    breadcrumb_col1, breadcrumb_col2 = st.columns([1, 4])
    with breadcrumb_col1:
        display_section_header("🔄 Version Selection & Feedback")
    
    # Make sure we have the session data we need
    if ('original_jd' not in st.session_state or 
        'enhanced_versions' not in st.session_state or 
        len(st.session_state.enhanced_versions) < 3):
        st.error("Please generate enhanced versions first before proceeding to refinement.")
        if st.button("Go to Generation Page", key="goto_gen"):
            switch_page("jd_enhance")
        return
    
    # Setup the layout with two columns
    left_col, right_col = st.columns([1, 1])
    
    # In the left column, show the version selection and previous feedback
    with left_col:
        display_subsection_header("1. Select Version")
        selected_version = st.radio(
            "Choose the version you'd like to use as a base:",
            ["Version 1", "Version 2", "Version 3"],
            help="Select the version that best matches your needs for further enhancement",
            key="version_selector"
        )
        
        # Get selected version index
        selected_index = int(selected_version[-1]) - 1
        
        # Display the selected version
        st.text_area(
            f"Selected Version Content",
            st.session_state.enhanced_versions[selected_index],
            height=250,
            disabled=True,
            key=f"selected_version_display"
        )
        
        # Display previous feedback if available
        if logger.current_state["feedback_history"]:
            display_subsection_header("Previous Feedback")
            with st.expander("View Previous Feedback", expanded=True):
                for i, feedback in enumerate(logger.current_state["feedback_history"], 1):
                    if isinstance(feedback, dict):
                        feedback_text = feedback.get("feedback", "")
                        feedback_type = feedback.get("type", "General Feedback")
                        st.markdown(f"**#{i} - {feedback_type}:**")
                        st.markdown(f"> {feedback_text}")
                        st.markdown("---")
                    else:
                        st.markdown(f"**#{i}:**")
                        st.markdown(f"> {feedback}")
                        st.markdown("---")
    
    # In the right column, show the feedback input and submission options
    with right_col:
        display_subsection_header("2. Provide Feedback")
        
        # Define feedback type options
        feedback_types = ["General Feedback", "Rejected Candidate", "Hiring Manager Feedback", 
                         "Client Feedback", "Selected Candidate", "Interview Feedback"]
        
        # Select feedback type
        selected_feedback_type = st.selectbox(
            "Feedback Type:",
            options=feedback_types,
            index=feedback_types.index(st.session_state.feedback_type) if st.session_state.feedback_type in feedback_types else 0,
            key="feedback_type_selector"
        )
        
        # Update feedback type if changed
        if selected_feedback_type != st.session_state.feedback_type:
            st.session_state.feedback_type = selected_feedback_type
        
        # Handle feedback clearing mechanism
        if st.session_state.get('clear_feedback', False):
            st.session_state.clear_feedback = False
        
        # Input for manual feedback
        user_feedback = st.text_area(
            "Enter your suggestions for improving the selected version:",
            height=150,
            placeholder="E.g., 'Add more emphasis on leadership skills', 'Include cloud technologies', 'Remove references to specific programming languages', etc.",
            key="user_feedback",
            help="Be specific about what you'd like to change or improve"
        )
        
        # Create a button to add manual feedback
        if st.button("Add Manual Feedback", type="secondary", key="add_feedback_only"):
            if user_feedback.strip():
                # Create feedback object with type
                feedback_obj = {
                    "feedback": user_feedback,
                    "type": selected_feedback_type,
                    "role": st.session_state.role
                }
                
                # Add to the logger's feedback history directly
                logger.current_state["feedback_history"].append(feedback_obj)
                
                # Save the updated state
                logger._save_state()
                
                # Add to session state for UI update
                if 'feedback_history' not in st.session_state:
                    st.session_state.feedback_history = []
                st.session_state.feedback_history.append(feedback_obj)
                
                # Log the action
                logger.current_state["actions"].append({
                    "action": "feedback",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "index": len(logger.current_state["feedback_history"]) - 1
                })
                logger._save_state()
                
                # Clear feedback input
                st.session_state.clear_feedback = True
                display_success_message("Feedback added successfully! You can add more feedback or generate the final version when ready.")
                st.rerun()
            else:
                display_warning_message("Please enter some feedback first.")
        
        # View all feedback button
        if st.button("View All Feedback", type="secondary", key="view_all_feedback"):
            st.session_state['viewing_all_feedback'] = True
    
    # Display all feedback if requested
    if st.session_state.get('viewing_all_feedback', False):
        display_section_header("📋 All Feedback History")
        display_filtered_feedback_history()
        # Reset viewing flag after displaying
        st.session_state['viewing_all_feedback'] = False
    
    # Final enhancement process section
    display_section_header("🚀 Generate Final Version")
    
    final_col1, final_col2 = st.columns(2)
        
    with final_col1:
        current_feedback_count = len(logger.current_state['feedback_history'])
        current_btn_label = f"Generate Final Version ({current_feedback_count} feedback items)"
        if st.button(current_btn_label, type="primary", key="generate_current_feedback"):
            try:
                with st.spinner("Creating final version with feedback... This may take a moment"):
                    # Log version selection
                    logger.log_version_selection(selected_index)
                        
                    # Log new feedback if provided and not already added
                    if user_feedback.strip():
                        # Create feedback object with type
                        feedback_obj = {
                            "feedback": user_feedback,
                            "type": selected_feedback_type,
                            "role": st.session_state.role
                        }
                        
                        # Add to the logger's feedback history directly
                        logger.current_state["feedback_history"].append(feedback_obj)
                        
                        # Save the updated state
                        logger._save_state()
                        
                        # Log the action
                        logger.current_state["actions"].append({
                            "action": "feedback",
                            "timestamp": datetime.datetime.now().isoformat(),
                            "index": len(logger.current_state["feedback_history"]) - 1
                        })
                        logger._save_state()
                        
                        # Clear feedback input
                        st.session_state.clear_feedback = True
                        
                    # Use the current enhanced version if it exists, otherwise use selected version
                    base_description = (logger.current_state["current_enhanced_version"] or 
                                        st.session_state.enhanced_versions[selected_index])
                        
                    # Generate final description using current session feedback
                    final_description = agent.generate_final_description(
                        base_description,
                        logger.current_state["feedback_history"]
                    )
                        
                    # Log the new enhanced version
                    logger.log_enhanced_version(final_description, is_final=True)
                        
                    # Store the new enhanced version in session state
                    st.session_state.current_enhanced_version = final_description
                        
                    # Flag that we generated the final version
                    st.session_state.final_version_generated = True
                    st.session_state.final_version = final_description
                        
                    display_success_message("Final version generated successfully!")
                    st.rerun()
                        
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error("Please try again or contact support if the problem persists.")
        
    with final_col2:
        st.caption("""
        Generate the final enhanced version based on your selected template and all feedback provided. 
        The AI will incorporate all feedback items to create an optimized job description that meets your requirements.
        """)
        
    # Display final version if it was generated
    if st.session_state.get('final_version_generated', False) and st.session_state.get('final_version'):
        final_description = st.session_state.final_version
            
        # Display final version with clear separation
        st.markdown("---")
        display_section_header("✅ Final Enhanced Job Description")
        
        # Create a container with a border for the final version
        final_container = st.container(border=True)
        with final_container:
            st.text_area(
                "Final Content",
                final_description,
                height=400,
                key="final_description"
            )
            
        # Add final version to scores dictionary for comparison
        original_scores = analyzer.analyze_text(st.session_state.original_jd)
        final_scores = analyzer.analyze_text(final_description)
        
        # Create comparison section
        display_section_header("📊 Final Analysis")
        
        final_col1, final_col2 = st.columns([1, 1])
            
        with final_col1:
            final_radar = create_multi_radar_chart({'Original': original_scores, 'Final': final_scores})
            st.plotly_chart(final_radar, use_container_width=True, key="final_radar")
            
        with final_col2:
            # Create a simplified comparison for final vs original
            final_comparison_df = create_comparison_dataframe({'Original': original_scores, 'Final': final_scores})
            st.dataframe(
                final_comparison_df,
                height=400,
                use_container_width=True,
                hide_index=True,
                key="final_comparison"
            )
            
        # Download section
        display_section_header("📥 Download Options")
        
        download_col1, download_col2 = st.columns(2)
            
        with download_col1:
            st.download_button(
                label="Download as TXT",
                data=final_description,
                file_name=f"enhanced_jd_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                key="download_txt"
            )
            # Log download action
            logger.log_download("txt", f"enhanced_jd_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            
        with download_col2:
            if st.button("Download as DOCX", key="download_docx"):
                docx_filename = f"enhanced_jd_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
                save_enhanced_jd(final_description, docx_filename, 'docx')
                display_success_message(f"Saved as {docx_filename}")
                # Log download action
                logger.log_download("docx", docx_filename)