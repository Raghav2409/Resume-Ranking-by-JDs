import streamlit as st
import datetime
import json
from utils.job_search import JobSearchUtility

class StateManager:
    """
    Unified state manager to maintain context across app tabs
    
    This class provides:
    1. Centralized state management
    2. Event notifications between tabs
    3. Persistent context for job descriptions, feedback, and analytics
    4. History tracking for state changes
    """
    
    def __init__(self):
        """Initialize the state manager with default values"""
        self.state_history = []
        self.last_update = datetime.datetime.now().isoformat()
        
        # Initialize job search utility if needed
        if 'job_search_utility' not in st.session_state:
            st.session_state['job_search_utility'] = JobSearchUtility()
    
    def get(self, key, default=None):
        """
        Get a value from session state with fallback default
        
        Args:
            key (str): Key to retrieve
            default: Default value if key doesn't exist
            
        Returns:
            Value from session state or default
        """
        # Check for special case keys that need special handling
        if key == 'job_search_utility' and 'job_search_utility' not in st.session_state:
            st.session_state['job_search_utility'] = JobSearchUtility()
        
        return st.session_state.get(key, default)
    
    def set(self, key, value, track_history=False, source_tab=None):
        """
        Set a value in session state with optional history tracking
        
        Args:
            key (str): Key to set
            value: Value to store
            track_history (bool): Whether to track this change in history
            source_tab (str): Tab that initiated this change
        """
        # Special handling for job search utility
        if key == 'job_search_utility' and isinstance(value, JobSearchUtility):
            # Ensure the value is properly initialized
            if value.is_initialized:
                st.session_state['job_search_initialized'] = True
        
        # Store in session state
        st.session_state[key] = value
        
        # Track in history if requested
        if track_history:
            self.state_history.append({
                'key': key,
                'timestamp': datetime.datetime.now().isoformat(),
                'source_tab': source_tab
            })
            self.last_update = datetime.datetime.now().isoformat()
    
    def ensure_job_search_initialization(self):
        """
        Ensure job search utility is properly initialized across tabs
        
        Returns:
            bool: True if job search is initialized, False otherwise
        """
        # Check if job search is already initialized
        if st.session_state.get('job_search_initialized', False):
            job_search = self.get('job_search_utility')
            
            # If job_search exists but isn't fully initialized, try to initialize it
            if job_search and not job_search.is_initialized:
                # Create dummy data for demonstration
                import pandas as pd
                
                # Create sample position report data
                job_search.position_report_df = pd.DataFrame({
                    'Parent Id': ['1001', '1002', '1003', '1004', '1005'],
                    'Job Description': [
                        'Software Engineer with 5+ years experience in Python and Java...',
                        'Data Scientist with strong background in machine learning...',
                        'DevOps Engineer with expertise in AWS and CI/CD pipelines...',
                        'Frontend Developer with React.js experience...',
                        'Backend Developer with Node.js and MongoDB experience...'
                    ]
                })
                
                # Create sample job listings data
                job_search.job_listings_df = pd.DataFrame({
                    'Job Id': ['1001', '1002', '1003', '1004', '1005'],
                    'Reference Id': ['REF1001', 'REF1002', 'REF1003', 'REF1004', 'REF1005'],
                    'Job Name': [
                        'Software Engineer', 
                        'Data Scientist', 
                        'DevOps Engineer', 
                        'Frontend Developer',
                        'Backend Developer'
                    ],
                    'Client': [
                        'TechCorp Inc.', 
                        'DataAnalytics Ltd.', 
                        'CloudSystems Inc.', 
                        'WebApp Solutions',
                        'ServerTech Inc.'
                    ]
                })
                
                # Set as initialized
                job_search.is_initialized = True
                self.set('job_search_utility', job_search)
            
            return True
            
        return False
    
    def update_jd_repository(self, key, value, source_tab=None):
        """
        Update a specific key in the JD repository
        
        Args:
            key (str): Repository key to update
            value: New value
            source_tab (str): Tab that initiated this change
        """
        jd_repository = self.get('jd_repository', {})
        jd_repository[key] = value
        
        # Store updated repository
        self.set('jd_repository', jd_repository, track_history=True, source_tab=source_tab)
        
        # Create notification for other tabs
        self.add_notification({
            'type': 'jd_updated',
            'key': key,
            'origin': source_tab,
            'timestamp': datetime.datetime.now().isoformat()
        })
    
    def update_feedback_repository(self, key, value, source_tab=None):
        """
        Update a specific key in the feedback repository
        
        Args:
            key (str): Repository key to update
            value: New value
            source_tab (str): Tab that initiated this change
        """
        feedback_repository = self.get('feedback_repository', {})
        feedback_repository[key] = value
        
        # Store updated repository
        self.set('feedback_repository', feedback_repository, track_history=True, source_tab=source_tab)
        
        # Create notification for feedback updates
        if key == 'history':
            self.add_notification({
                'type': 'feedback_added',
                'origin': source_tab,
                'timestamp': datetime.datetime.now().isoformat()
            })
    
    def add_notification(self, notification):
        """
        Add a notification to the notification bus
        
        Args:
            notification (dict): Notification details
        """
        notifications = self.get('notifications', [])
        notifications.append(notification)
        self.set('notifications', notifications)
    
    def clear_notifications(self):
        """Clear all pending notifications"""
        self.set('notifications', [])
    
    def get_jd_content(self):
        """
        Get the currently active job description content
        
        Returns:
            tuple: (content, source_name, unique_id)
        """
        jd_repository = self.get('jd_repository', {})
        
        # Check for final version first
        if jd_repository.get('final_version'):
            return (
                jd_repository.get('final_version'),
                f"Final Version of {jd_repository.get('source_name', 'Job Description')}",
                jd_repository.get('unique_id')
            )
        
        # Check for selected enhanced version
        if jd_repository.get('enhanced_versions') and len(jd_repository.get('enhanced_versions')) > 0:
            selected_idx = jd_repository.get('selected_version_idx', 0)
            versions = jd_repository.get('enhanced_versions')
            if 0 <= selected_idx < len(versions):
                return (
                    versions[selected_idx],
                    f"Enhanced Version {selected_idx + 1} of {jd_repository.get('source_name', 'Job Description')}",
                    jd_repository.get('unique_id')
                )
        
        # Fall back to original
        return (
            jd_repository.get('original'),
            jd_repository.get('source_name'),
            jd_repository.get('unique_id')
        )
    
    def has_active_jd(self):
        """Check if there's an active job description in the repository"""
        jd_repository = self.get('jd_repository', {})
        return jd_repository.get('original') is not None
    
    def export_state(self):
        """
        Export the current state as JSON
        
        Returns:
            str: JSON representation of state
        """
        export_data = {
            'jd_repository': self.get('jd_repository', {}),
            'feedback_repository': self.get('feedback_repository', {}),
            'analytics_repository': self.get('analytics_repository', {}),
            'state_history': self.state_history,
            'last_update': self.last_update,
            'session_id': self.get('session_id'),
            'role': self.get('role'),
            'job_search_initialized': self.get('job_search_initialized', False)
        }
        
        return json.dumps(export_data, indent=2)
    
    def import_state(self, state_json):
        """
        Import state from JSON
        
        Args:
            state_json (str): JSON representation of state
            
        Returns:
            bool: Success or failure
        """
        try:
            import_data = json.loads(state_json)
            
            # Update repositories
            if 'jd_repository' in import_data:
                self.set('jd_repository', import_data['jd_repository'])
                
            if 'feedback_repository' in import_data:
                self.set('feedback_repository', import_data['feedback_repository'])
                
            if 'analytics_repository' in import_data:
                self.set('analytics_repository', import_data['analytics_repository'])
                
            # Update history and metadata
            if 'state_history' in import_data:
                self.state_history = import_data['state_history']
                
            if 'last_update' in import_data:
                self.last_update = import_data['last_update']
                
            if 'session_id' in import_data:
                self.set('session_id', import_data['session_id'])
                
            if 'role' in import_data:
                self.set('role', import_data['role'])
                
            if 'job_search_initialized' in import_data:
                self.set('job_search_initialized', import_data['job_search_initialized'])
            
            return True
        except Exception as e:
            print(f"Error importing state: {e}")
            return False
    
    def handle_jd_selection(self, jd_content, jd_source_name, jd_unique_id, context=""):
        """
        Handle job description selection with proper state updates
        
        Args:
            jd_content (str): Job description content
            jd_source_name (str): Source name of the job description
            jd_unique_id (str): Unique ID of the job description
            context (str): Context where this function is called
            
        Returns:
            bool: Success or failure
        """
        if not jd_content or not jd_source_name or not jd_unique_id:
            return False
            
        # Update the JD repository
        self.update_jd_repository('original', jd_content, source_tab=context)
        self.update_jd_repository('source_name', jd_source_name, source_tab=context)
        self.update_jd_repository('unique_id', jd_unique_id, source_tab=context)
        
        # Reset versions when changing JD source
        self.update_jd_repository('enhanced_versions', [], source_tab=context)
        self.update_jd_repository('selected_version_idx', 0, source_tab=context)
        self.update_jd_repository('final_version', None, source_tab=context)
        
        # Add notification for JD selection
        self.add_notification({
            'type': 'jd_selected',
            'source_name': jd_source_name,
            'origin': context,
            'timestamp': datetime.datetime.now().isoformat()
        })
        
        return True