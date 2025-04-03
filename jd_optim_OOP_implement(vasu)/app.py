import streamlit as st
import os
import datetime
import uuid

# Import components
from config import set_page_config, custom_css
from utils.file_utils import save_enhanced_jd
from utils.job_search import JobSearchUtility
from models.job_description_analyzer import JobDescriptionAnalyzer
from models.job_description_agent import JobDescriptionAgent
from jdoptim_logger import JDOptimLogger
from state_manager import StateManager

# Import UI components
from ui.common import render_header, render_role_selector, render_tabs
from ui.jd_optimization import render_jd_optimization_page
from ui.candidate_ranking import render_candidate_ranking_page
from ui.interview_prep import render_interview_prep_page
from ui.client_feedback import render_client_feedback_page

def init_session_state():
    """Initialize global session state manager"""
    if 'state_manager' not in st.session_state:
        # Create state manager with built-in defaults
        st.session_state.state_manager = StateManager()
        state_manager = st.session_state.state_manager
        
        # Initialize essential global state
        state_manager.set('session_id', str(uuid.uuid4()))
        state_manager.set('role', 'Recruiter')
        state_manager.set('active_tab', "JD Optimization")
        
        # Initialize empty containers for job descriptions
        state_manager.set('jd_repository', {
            'original': None,           # Original JD content
            'source_name': None,        # Name/source of JD
            'unique_id': None,          # Unique ID for caching
            'enhanced_versions': [],    # List of enhanced versions 
            'selected_version_idx': 0,  # User-selected version
            'final_version': None,      # Final enhanced version
        })
        
        # Initialize feedback repository
        state_manager.set('feedback_repository', {
            'history': [],              # All feedback items
            'current_feedback': '',     # Current feedback text
            'current_type': 'General Feedback'  # Current feedback type
        })
        
        # Initialize analytics repository
        state_manager.set('analytics_repository', {
            'original_scores': None,
            'version_scores': {},
            'final_scores': None
        })
        
        # Initialize resume repository
        state_manager.set('resume_repository', {
            'pools': [],
            'ranked_candidates': [],
            'analysis_results': None
        })
        
        # Register notification bus
        state_manager.set('notifications', [])
        
        # Initialize job search utility
        state_manager.set('job_search_utility', JobSearchUtility())
        state_manager.set('job_search_initialized', False)

def get_or_create_logger():
    """Get existing logger from session state or create a new one with proper integration"""
    state_manager = st.session_state.state_manager
    
    # First check if we have a logger in session state
    if 'logger' in st.session_state:
        logger = st.session_state.logger
        
        # Ensure logger has latest role
        if logger.username != state_manager.get('role'):
            logger.username = state_manager.get('role')
            logger.current_state["username"] = state_manager.get('role')
            logger._save_state()
            
        return logger
    
    # Try to load existing session by ID
    session_id = state_manager.get('session_id')
    try:
        logger = JDOptimLogger.load_session(session_id)
        if logger:
            # Update role if it changed
            if logger.username != state_manager.get('role'):
                logger.username = state_manager.get('role')
                logger.current_state["username"] = state_manager.get('role')
                logger._save_state()
            
            st.session_state.logger = logger
            return logger
    except Exception as e:
        print(f"Failed to load existing session: {e}")
    
    # Create a new logger with the current role
    logger = JDOptimLogger(username=state_manager.get('role'))
    
    # Update session ID in state manager
    state_manager.set('session_id', logger.session_id)
    st.session_state.logger = logger
    
    return logger

def main():
    """Main function with improved integration between app components"""
    # Configure the Streamlit page
    set_page_config()

    # Apply custom CSS
    st.markdown(custom_css, unsafe_allow_html=True)

    # Initialize session state
    init_session_state()
    
    # Get state manager and logger
    state_manager = st.session_state.state_manager
    logger = get_or_create_logger()

    # Render header with logo and title
    render_header()
    
    # Render role selector (updates state manager)
    render_role_selector(state_manager)
    
    # Render navigation tabs (updates state manager)
    render_tabs(state_manager)
    
    # Initialize the analyzer and agent as shared services
    analyzer = JobDescriptionAnalyzer()
    agent = JobDescriptionAgent(model_id="anthropic.claude-3-haiku-20240307-v1:0")
    
    # Create service container for shared resources
    services = {
        'logger': logger,
        'analyzer': analyzer,
        'agent': agent,
        'state_manager': state_manager
    }
    
    # Check for notifications from other tabs
    process_notifications(state_manager)
    
    # Render the appropriate page based on active tab
    active_tab = state_manager.get('active_tab')
    if active_tab == "JD Optimization":
        render_jd_optimization_page(services)
    elif active_tab == "Candidate Ranking":
        render_candidate_ranking_page(services)
    elif active_tab == "Client Feedback":
        render_client_feedback_page(services)
    elif active_tab == "Interview Prep":
        render_interview_prep_page()
    
    # Footer with company info
    st.markdown("---")
    footer_col1, footer_col2 = st.columns([4, 1])
    
    with footer_col1:
        st.caption("JD Agent | Made by Apexon")
    
    with footer_col2:
        st.caption(f"v2.0 - {datetime.datetime.now().strftime('%Y')}")

def process_notifications(state_manager):
    """Process any pending notifications between tabs"""
    notifications = state_manager.get('notifications', [])
    
    if notifications:
        # Process each notification
        for notification in notifications:
            notify_type = notification.get('type')
            
            if notify_type == 'jd_selected':
                # Job description was selected in another tab
                st.success(f"Using job description: {notification.get('source_name')}")
            
            elif notify_type == 'feedback_added':
                # Feedback was added in another tab
                st.info(f"New feedback added in {notification.get('origin')} tab")
            
            elif notify_type == 'version_enhanced':
                # Job description was enhanced in another tab
                st.success(f"Job description enhanced in {notification.get('origin')} tab")
        
        # Clear notifications after processing
        state_manager.set('notifications', [])

if __name__ == "__main__":
    main()