import streamlit as st
import os
import datetime
import uuid

# Import components
from config import set_page_config, custom_css
from utils.file_utils import read_job_description, save_enhanced_jd
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
    else:
        # Ensure job search utility is properly initialized
        state_manager = st.session_state.state_manager
        state_manager.ensure_job_search_initialization()

class MockAnalyzer:
    """Mock analyzer for when the real one is not available"""
    def analyze_text(self, text):
        """Mock analysis function"""
        return {
            'Technical Skills': 0.7,
            'Soft Skills': 0.6,
            'Experience Level': 0.8,
            'Education': 0.5,
            'Tools & Technologies': 0.7,
            'Domain Knowledge': 0.6
        }

class MockAgent:
    """Mock agent for when the real one is not available"""
    def __init__(self, model_id="mock-model"):
        self.model_id = model_id
    
    def generate_initial_descriptions(self, job_description):
        """Generate enhanced JD versions"""
        return [
            f"Enhanced Version 1:\n\n{job_description}\n\nAdditional skills: Python, Java, etc.",
            f"Enhanced Version 2:\n\n{job_description}\n\nAdditional requirements: 5+ years experience",
            f"Enhanced Version 3:\n\n{job_description}\n\nAdditional qualifications: Bachelor's degree"
        ]
    
    def generate_final_description(self, selected_description, feedback_history):
        """Generate final enhanced description"""
        return f"{selected_description}\n\nIncorporated feedback: " + "\n".join([
            f"- {f.get('feedback', f) if isinstance(f, dict) else f}"
            for f in feedback_history[-2:] if f  # Only include last 2 feedbacks for brevity
        ])

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

def initialize_services():
    """Initialize and return service container with shared resources"""
    try:
        # Get or create logger
        logger = get_or_create_logger()
        
        # Initialize analyzer
        try:
            analyzer = JobDescriptionAnalyzer()
        except Exception as e:
            st.error(f"Error initializing analyzer: {e}")
            analyzer = MockAnalyzer()
        
        # Initialize agent
        try:
            agent = JobDescriptionAgent(model_id="anthropic.claude-3-haiku-20240307-v1:0")
        except Exception as e:
            st.error(f"Error initializing AI agent: {e}")
            agent = MockAgent()
        
        # Get state manager
        state_manager = st.session_state.state_manager
        
        # Create service container
        services = {
            'logger': logger,
            'analyzer': analyzer,
            'agent': agent,
            'state_manager': state_manager
        }
        
        return services
    except Exception as e:
        st.error(f"Error initializing services: {e}")
        
        # Provide fallback services
        return {
            'logger': None,
            'analyzer': MockAnalyzer(),
            'agent': MockAgent(),
            'state_manager': st.session_state.state_manager
        }

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

def main():
    """Main function with improved integration between app components"""
    # Configure the Streamlit page
    try:
        set_page_config()
    except Exception as e:
        st.set_page_config(
            page_title="JD Agent",
            page_icon="💼",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
        st.warning(f"Error setting page config: {e}")

    # Apply custom CSS
    try:
        st.markdown(custom_css, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Error applying custom CSS: {e}")
    
    # Initialize session state
    try:
        init_session_state()
    except Exception as e:
        st.error(f"Error initializing session state: {e}")
        # Create minimal state manager if initialization fails
        if 'state_manager' not in st.session_state:
            st.session_state.state_manager = StateManager()
    
    # Get state manager
    state_manager = st.session_state.state_manager
    
    # Initialize services
    services = initialize_services()

    # Render header with logo and title
    try:
        render_header()
    except Exception as e:
        st.title("Dynamic Job Description Optimizer")
        st.warning(f"Error rendering header: {e}")
    
    # Render role selector (updates state manager)
    try:
        render_role_selector(state_manager)
    except Exception as e:
        st.warning(f"Error rendering role selector: {e}")
        # Simple fallback role selector
        st.selectbox("Your Role:", ["Recruiter", "Hiring Manager", "Candidate"], key="simple_role")
    
    # Render navigation tabs (updates state manager)
    try:
        render_tabs(state_manager)
    except Exception as e:
        st.warning(f"Error rendering tabs: {e}")
        # Simple fallback tabs
        tabs = st.tabs(["JD Optimization", "Candidate Ranking", "Client Feedback", "Interview Prep"])
        active_tab = st.session_state.get('active_tab', "JD Optimization")
    
    # Check for notifications from other tabs
    process_notifications(state_manager)
    
    # Render the appropriate page based on active tab
    try:
        active_tab = state_manager.get('active_tab')
        if active_tab == "JD Optimization":
            render_jd_optimization_page(services)
        elif active_tab == "Candidate Ranking":
            render_candidate_ranking_page(services)
        elif active_tab == "Client Feedback":
            render_client_feedback_page(services)
        elif active_tab == "Interview Prep":
            render_interview_prep_page(services)
    except Exception as e:
        st.error(f"Error rendering tab content: {e}")
        st.error("Please refresh the page and try again.")
    
    # Footer with company info
    st.markdown("---")
    footer_col1, footer_col2 = st.columns([4, 1])
    
    with footer_col1:
        st.caption("JD Agent | Made by Apexon")
    
    with footer_col2:
        st.caption(f"v2.0 - {datetime.datetime.now().strftime('%Y')}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Critical error: {e}")
        st.error("Please refresh the page to restart the application.")