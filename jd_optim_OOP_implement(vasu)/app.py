import streamlit as st
import os
import datetime
import uuid

# Import components
from config import set_page_config, custom_css
from utils.file_utils import save_enhanced_jd
from models.job_description_analyzer import JobDescriptionAnalyzer
from models.job_description_agent import JobDescriptionAgent
from jdoptim_logger import JDOptimLogger

# Import UI components
from ui.common import render_header, render_role_selector, render_tabs, switch_tab, switch_page
from ui.jd_optimization import render_jd_optimization_page  # New unified page
from ui.candidate_ranking import render_candidate_ranking_page
from ui.interview_prep import render_interview_prep_page
from ui.client_feedback import render_client_feedback_page

def init_session_state():
    """Initialize session state variables if they don't exist"""
    if 'role' not in st.session_state:
        st.session_state.role = 'Recruiter'  # Default role
    
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        
    if 'feedback_history' not in st.session_state:
        st.session_state.feedback_history = []
        
    if 'last_file' not in st.session_state:
        st.session_state.last_file = None
        
    if 'reload_flag' not in st.session_state:
        st.session_state.reload_flag = False
        
    if 'clear_feedback' not in st.session_state:
        st.session_state.clear_feedback = False
        
    if 'viewing_all_feedback' not in st.session_state:
        st.session_state.viewing_all_feedback = False
        
    if 'viewing_session_feedback' not in st.session_state:
        st.session_state.viewing_session_feedback = False
        
    if 'final_version_generated' not in st.session_state:
        st.session_state.final_version_generated = False
        
    if 'final_version' not in st.session_state:
        st.session_state.final_version = None
        
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "jd_enhance"  # Default page
        
    if 'feedback_type' not in st.session_state:
        st.session_state.feedback_type = "General Feedback"
        
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "JD Optimization"
        
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
        
    if 'ranked_candidates' not in st.session_state:
        st.session_state.ranked_candidates = []
        
    # Resume pool handling
    if 'resume_pools' not in st.session_state:
        st.session_state.resume_pools = []
        
    # Client feedback tab specific state
    if 'client_jd' not in st.session_state:
        st.session_state.client_jd = None
        
    if 'client_feedback' not in st.session_state:
        st.session_state.client_feedback = None
        
    if 'client_feedback_type' not in st.session_state:
        st.session_state.client_feedback_type = "Client Feedback"
        
    if 'client_enhanced_jd' not in st.session_state:
        st.session_state.client_enhanced_jd = None

def get_or_create_logger():
    """Get existing logger from session state or create a new one"""
    # First check if we have a logger in session state
    if 'logger' in st.session_state:
        return st.session_state.logger
    
    # If we have a session_id, try to load that session
    if 'session_id' in st.session_state:
        try:
            # Try to load existing session by ID
            logger = JDOptimLogger.load_session(st.session_state.session_id)
            if logger:
                # Update role if it changed
                if logger.username != st.session_state.role:
                    logger.username = st.session_state.role
                    logger.current_state["username"] = st.session_state.role
                    logger._save_state()
                
                st.session_state.logger = logger
                return logger
        except Exception as e:
            # If loading fails, we'll create a new logger below
            print(f"Failed to load existing session: {e}")
    
    # Create a new logger with the current role
    logger = JDOptimLogger(username=st.session_state.role)
    st.session_state.session_id = logger.session_id
    st.session_state.logger = logger
    
    return logger

def main():
    """Main function to run the JD Agent application"""
    # Configure the Streamlit page
    set_page_config()

    # Apply custom CSS
    st.markdown(custom_css, unsafe_allow_html=True)

    # Initialize session state
    init_session_state()
    
    # Get or create logger
    logger = get_or_create_logger()

    # Render header with logo and title
    render_header()
    
    # Render role selector
    render_role_selector()
    
    # Render navigation tabs
    render_tabs()
    
    # Initialize the analyzer and agent
    analyzer = JobDescriptionAnalyzer()
    agent = JobDescriptionAgent(model_id="anthropic.claude-3-haiku-20240307-v1:0")
    
    # Render the appropriate page based on active tab
    if st.session_state.active_tab == "JD Optimization":
        render_jd_optimization_page(logger, analyzer, agent)
    elif st.session_state.active_tab == "Candidate Ranking":
        render_candidate_ranking_page()
    elif st.session_state.active_tab == "Client Feedback":
        render_client_feedback_page(logger, analyzer, agent)
    elif st.session_state.active_tab == "Interview Prep":
        render_interview_prep_page()
    
    # Footer with company info
    st.markdown("---")
    footer_col1, footer_col2 = st.columns([4, 1])
    
    with footer_col1:
        st.caption("JD Agent | Made by Apexon")
    
    with footer_col2:
        st.caption(f"v2.0 - {datetime.datetime.now().strftime('%Y')}")

if __name__ == "__main__":
    main()