import streamlit as st

def render_header():
    """Render the application header with logo and title"""
    header_col1, header_col2, header_col3 = st.columns([1, 3, 1])
    
    with header_col1:
        st.markdown(
            """
            <div style="display: flex; justify-content: center; align-items: center; height: 60px;">
                <img src="https://img.icons8.com/color/96/000000/briefcase.png" alt="Dynamic Job Description Optimizer" width="50" height="50">
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with header_col2:
        st.markdown("<h1 style='text-align: center; margin: 0;'>Dynamic Job Description Optimizer</h1>", unsafe_allow_html=True)
    
    with header_col3:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center; align-items: center; height: 60px;">
                <div style="padding: 5px 10px; border-radius: 5px; text-align: center;">
                    <div style="font-weight: bold;">{st.session_state.role}</div>
                    <div style="font-size: 0.8em;">View</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

def render_role_selector():
    """Render the role selector in a compact layout"""
    # Display role selector in a small container with border
    with st.container(border=True):
        # Define available roles
        roles = ["Recruiter", "Hiring Manager", "Candidate", "HR Manager", "Team Lead"]
        
        # Simple one-row layout
        selected_role = st.selectbox(
            "Your Role:",
            options=roles,
            index=roles.index(st.session_state.role) if st.session_state.role in roles else 0,
            help="Select your role in the hiring process"
        )
        
        # Update role if changed
        if selected_role != st.session_state.role:
            st.session_state.role = selected_role
            
            # Update logger if it exists
            if 'logger' in st.session_state:
                st.session_state.logger.username = selected_role
                st.session_state.logger.current_state["username"] = selected_role
                st.session_state.logger._save_state()

def render_tabs():
    """Render the navigation tabs"""
    # Updated tab names to match new UI
    tabs = ["JD Optimization", "Candidate Ranking", "Client Feedback", "Interview Prep"]
    
    # Create tab buttons in a row
    cols = st.columns(len(tabs))
    
    for i, tab in enumerate(tabs):
        with cols[i]:
            is_active = st.session_state.active_tab == tab
            bg_color = "#DBEAFE" if is_active else "#F9FAFB"
            text_color = "#1E40AF" if is_active else "#374151"
            border_bottom = "3px solid #2563EB" if is_active else "1px solid #E5E7EB"
            
            # Create button with styling that looks like a tab
            if st.button(
                tab,
                key=f"tab_{tab}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
                help=f"Switch to {tab} tab"
            ):
                switch_tab(tab)

def switch_tab(tab_name):
    """Switch between tabs in the application"""
    st.session_state.active_tab = tab_name
    
def switch_page(page_name):
    """Switch between main pages in the application"""
    st.session_state.current_page = page_name

def display_success_message(message):
    """Display a success message"""
    st.markdown(f"""
    <div class="success-box">
        ✅ {message}
    </div>
    """, unsafe_allow_html=True)
    
def display_warning_message(message):
    """Display a warning message"""
    st.markdown(f"""
    <div class="warning-box">
        ⚠️ {message}
    </div>
    """, unsafe_allow_html=True)
    
def display_info_message(message):
    """Display an info message"""
    st.markdown(f"""
    <div class="highlight-box">
        ℹ️ {message}
    </div>
    """, unsafe_allow_html=True)

def display_section_header(title):
    """Display a section header"""
    st.markdown(f"""<div class="section-header">{title}</div>""", unsafe_allow_html=True)

def display_subsection_header(title):
    """Display a subsection header"""
    st.markdown(f"""<div class="subsection-header">{title}</div>""", unsafe_allow_html=True)

def display_filtered_feedback_history(logger):
    """Display feedback history with filtering options"""
    # Get all available sessions
    sessions = logger.list_sessions()
    
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
    import datetime
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