import streamlit as st
import os
import datetime
from utils.file_utils import read_job_description

def render_header():
    """Render the application header with logo, title, and context info"""
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
        state_manager = st.session_state.state_manager
        jd_repository = state_manager.get('jd_repository', {})
        
        # Display contextual info if we have an active JD
        if jd_repository.get('source_name'):
            jd_type = "Original"
            
            if jd_repository.get('final_version'):
                jd_type = "Final Enhanced"
            elif jd_repository.get('enhanced_versions') and len(jd_repository.get('enhanced_versions')) > 0:
                jd_type = f"Enhanced v{jd_repository.get('selected_version_idx', 0) + 1}"
            
            st.markdown(
                f"""
                <div style="display: flex; justify-content: center; align-items: center; height: 60px;">
                    <div style="padding: 5px 10px; border-radius: 5px; text-align: center; 
                          background-color: rgba(66, 153, 225, 0.15); border: 1px solid #4299E1;">
                        <div style="font-weight: bold;">{state_manager.get('role')}</div>
                        <div style="font-size: 0.8em;" title="{jd_repository.get('source_name')}">{jd_type} JD Active</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="display: flex; justify-content: center; align-items: center; height: 60px;">
                    <div style="padding: 5px 10px; border-radius: 5px; text-align: center;">
                        <div style="font-weight: bold;">{state_manager.get('role')}</div>
                        <div style="font-size: 0.8em;">No Active JD</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

def render_role_selector(state_manager):
    """Render the role selector with state manager integration"""
    with st.container(border=True):
        # Define available roles
        roles = ["Recruiter", "Hiring Manager", "Candidate", "HR Manager", "Team Lead"]
        
        # Simple one-row layout
        selected_role = st.selectbox(
            "Your Role:",
            options=roles,
            index=roles.index(state_manager.get('role')) if state_manager.get('role') in roles else 0,
            help="Select your role in the hiring process"
        )
        
        # Update role if changed
        if selected_role != state_manager.get('role'):
            state_manager.set('role', selected_role)
            
            # Update logger if it exists
            if 'logger' in st.session_state:
                st.session_state.logger.username = selected_role
                st.session_state.logger.current_state["username"] = selected_role
                st.session_state.logger._save_state()

def render_tabs(state_manager):
    """Render the navigation tabs with state manager integration"""
    # Updated tab names to match new UI
    tabs = ["JD Optimization", "Candidate Ranking", "Client Feedback", "Interview Prep"]
    
    # Create tab buttons in a row
    cols = st.columns(len(tabs))
    
    for i, tab in enumerate(tabs):
        with cols[i]:
            is_active = state_manager.get('active_tab') == tab
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
                switch_tab(tab, state_manager)

def switch_tab(tab_name, state_manager):
    """Switch between tabs with state manager integration"""
    state_manager.set('active_tab', tab_name)
    
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



def render_feedback_component(state_manager, services, context=""):
    """
    Unified feedback collection component
    
    Provides consistent feedback collection across tabs with contextual awareness.
    
    Args:
        state_manager: The global state manager
        services (dict): Dictionary of services
        context (str): Context of where this component is being used
    """
    logger = services.get('logger')
    
    # Get feedback repository
    feedback_repository = state_manager.get('feedback_repository', {})
    
    # Define feedback types
    feedback_types = [
        "General Feedback", 
        "Rejected Candidate", 
        "Hiring Manager Feedback", 
        "Client Feedback", 
        "Selected Candidate", 
        "Interview Feedback"
    ]
    
    current_type = feedback_repository.get('current_type', "General Feedback")
    
    # Create a two-column layout for feedback
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Show previous feedback if available
        if feedback_repository.get('history'):
            st.markdown("**Previous Feedback:**")
            with st.expander("View Feedback History", expanded=False):
                for i, feedback in enumerate(feedback_repository.get('history'), 1):
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
    
    with col2:
        # Feedback type selection
        selected_feedback_type = st.selectbox(
            "Feedback Type:",
            options=feedback_types,
            index=feedback_types.index(current_type) if current_type in feedback_types else 0,
            key=f"{context}_feedback_type_selector"
        )
        
        # Update type in state if changed
        if selected_feedback_type != current_type:
            state_manager.update_feedback_repository('current_type', selected_feedback_type, source_tab=context)
        
        # Feedback input
        user_feedback = st.text_area(
            "Enter your feedback or suggestions:",
            height=150,
            value=feedback_repository.get('current_feedback', ''),
            placeholder="E.g., 'Add more emphasis on leadership skills', 'Include cloud technologies', etc.",
            key=f"{context}_feedback_input",
            help="Be specific about what you'd like to change or improve"
        )
        
        # Update current feedback in state
        if user_feedback != feedback_repository.get('current_feedback', ''):
            state_manager.update_feedback_repository('current_feedback', user_feedback, source_tab=context)
        
        # Add feedback button
        if st.button("➕ Add Feedback", type="secondary", key=f"{context}_add_feedback_btn"):
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
                state_manager.update_feedback_repository('history', history, source_tab=context)
                
                # Clear current feedback
                state_manager.update_feedback_repository('current_feedback', '', source_tab=context)
                
                # Log to logger
                if logger:
                    logger.log_feedback(user_feedback, selected_feedback_type)
                
                # Display success message
                display_success_message("Feedback added successfully!")
                
                # Force refresh
                st.rerun()
            else:
                display_warning_message("Please enter some feedback first.")

def display_jd_comparison(original_jd, enhanced_jd, services, context=""):
    """
    Display a comparison between original and enhanced JD
    
    Args:
        original_jd (str): Original JD content
        enhanced_jd (str): Enhanced JD content
        services (dict): Dictionary of services
        context (str): Context where this component is used
    """
    analyzer = services.get('analyzer')
    
    col1, col2 = st.columns(2)
    
    with col1:
        display_subsection_header("Original Job Description")
        st.text_area(
            "Original Content",
            original_jd,
            height=300,
            disabled=True,
            key=f"{context}_original_jd_display"
        )
    
    with col2:
        display_subsection_header("Enhanced Job Description")
        st.text_area(
            "Enhanced Content",
            enhanced_jd,
            height=300,
            key=f"{context}_enhanced_jd_display"
        )
    
    # Compare original vs enhanced with skill analysis
    if original_jd and enhanced_jd and analyzer:
        display_section_header("Comparison Analysis")
        
        # Analyze both versions
        original_scores = analyzer.analyze_text(original_jd)
        enhanced_scores = analyzer.analyze_text(enhanced_jd)
        
        comp_col1, comp_col2 = st.columns([1, 1])
        
        with comp_col1:
            display_subsection_header("Skill Coverage Comparison")
            from utils.visualization import create_multi_radar_chart
            radar_chart = create_multi_radar_chart({'Original': original_scores, 'Enhanced': enhanced_scores})
            st.plotly_chart(radar_chart, use_container_width=True, key=f"{context}_radar")
        
        with comp_col2:
            display_subsection_header("Detailed Analysis")
            from utils.visualization import create_comparison_dataframe
            comparison_df = create_comparison_dataframe({'Original': original_scores, 'Enhanced': enhanced_scores})
            st.dataframe(
                comparison_df,
                height=400,
                use_container_width=True,
                hide_index=True,
                key=f"{context}_comparison"
            )
            st.caption("Percentages indicate keyword coverage in each category")
            
def render_jd_selector(state_manager, services, context=""):
    """
    Unified job description selector component
    
    Provides consistent JD selection across tabs with contextual awareness.
    
    Args:
        state_manager: The global state manager
        services (dict): Dictionary of services
        context (str): Context of where this selector is being used
    
    Returns:
        bool: Whether a JD is selected
    """
    logger = services.get('logger')
    
    # Check if we already have an active JD
    jd_repository = state_manager.get('jd_repository', {})
    if jd_repository.get('original') is not None:
        # Show info about active JD
        source_name = jd_repository.get('source_name', 'Unknown')
        
        st.info(f"Currently using: {source_name}")
        
        # Option to change the JD
        change_jd = st.checkbox("Select a different job description", value=False, key=f"{context}_change_jd")
        
        if not change_jd:
            return True
    
    # Create a selection for the source type
    source_options = ["📁 File Selection", "📤 Upload New", "🔍 Search Database"]
    
    # Display selector for choosing JD source
    selected_source = st.radio(
        "Choose job description source:",
        options=source_options,
        horizontal=True,
        key=f"{context}_jd_source_selector"
    )
    
    # Variables to track job description source
    jd_content = None
    jd_source_name = None
    jd_unique_id = None
    
    # Handle each source option
    if selected_source == "🔍 Search Database":
        # First ensure job search is initialized
        job_search_initialized = render_job_search_section(state_manager)
        
        if not job_search_initialized:
            st.warning("Please initialize the job search database first.")
            return False
        
        # Get the job search utility
        job_search = state_manager.get('job_search_utility')
        
        # Get dropdown options
        options = job_search.get_dropdown_options()
        
        if not options:
            st.warning("No job listings found in the data.")
            return False
        
        # Add a search box for filtering the dropdown
        search_term = st.text_input("Search for job by ID, name, or client:", key=f"{context}_job_search_term")
        
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
                key=f"{context}_job_search_dropdown"
            )
            
            # Find and display the job description
            if selected_option:
                job_description, job_details = job_search.find_job_description(selected_option)
                
                if job_description:
                    jd_content = job_description
                    jd_source_name = selected_option
                    jd_unique_id = f"db_{job_details.get('Job Id', '')}"
                else:
                    st.error("Could not find job description for the selected job.")
                    return False
        else:
            st.warning("No jobs match your search criteria.")
            return False
    
    elif selected_source == "📁 File Selection":
        jd_directory = os.path.join(os.getcwd(), "Data/JDs")
        try:
            files = [f for f in os.listdir(jd_directory) if f.endswith(('.txt', '.docx'))]
            
            if files:
                selected_file = st.selectbox(
                    "Select Job Description File", 
                    files, 
                    key=f"{context}_file_selector"
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
                        return False
            else:
                st.warning("No job description files found in JDs directory. Creating sample files...")
                
                # Create the directory if it doesn't exist
                os.makedirs(jd_directory, exist_ok=True)
                
                # Write sample files
                for filename, content in sample_jds.items():
                    file_path = os.path.join(jd_directory, filename)
                    with open(file_path, "w") as f:
                        f.write(content)

        except FileNotFoundError:
            st.info("Directory 'JDs' not found. Creating it...")
            os.makedirs(jd_directory, exist_ok=True)
            st.rerun()
            return False
    
    elif selected_source == "📤 Upload New":
        uploaded_file = st.file_uploader(
            "Upload Job Description File", 
            type=['txt', 'docx'],
            key=f"{context}_file_uploader"
        )
        
        if uploaded_file:
            # Process uploaded file
            try:
                if uploaded_file.name.endswith('.txt'):
                    file_content = uploaded_file.getvalue().decode('utf-8')
                else:  # .docx
                    file_content = process_uploaded_docx(uploaded_file)
                
                jd_content = file_content
                jd_source_name = uploaded_file.name
                jd_unique_id = f"upload_{uploaded_file.name}"
            except Exception as e:
                st.error(f"Error processing uploaded file: {str(e)}")
                return False
        else:
            return False
    
    # Update state with the selected JD
    if jd_content and jd_source_name and jd_unique_id:
        # Update the JD repository
        state_manager.update_jd_repository('original', jd_content, source_tab=context)
        state_manager.update_jd_repository('source_name', jd_source_name, source_tab=context)
        state_manager.update_jd_repository('unique_id', jd_unique_id, source_tab=context)
        
        # Reset versions when changing JD source
        state_manager.update_jd_repository('enhanced_versions', [], source_tab=context)
        state_manager.update_jd_repository('selected_version_idx', 0, source_tab=context)
        state_manager.update_jd_repository('final_version', None, source_tab=context)
        
        # Add notification for JD selection
        state_manager.add_notification({
            'type': 'jd_selected',
            'source_name': jd_source_name,
            'origin': context,
            'timestamp': datetime.datetime.now().isoformat()
        })
        
        # Log file selection
        if logger:
            logger.log_file_selection(jd_source_name, jd_content)
        
        # Display success message
        display_success_message(f"Selected job description: {jd_source_name}")
        
        # Show JD preview
        with st.expander("View Job Description", expanded=True):
            st.text_area(
                "Content", 
                jd_content, 
                height=250, 
                disabled=True,
                key=f"{context}_jd_preview"
            )
        
        return True
    
    return False

# Add mock functions for process_uploaded_docx and display_success_message
# In case they're missing in the actual code
def process_uploaded_docx(uploaded_file):
    """Process an uploaded docx file (mock function)"""
    return "This is a placeholder for the document content"

def display_success_message(message):
    """Display a success message"""
    st.success(message)