import os
import json
import datetime
import uuid

class JDOptimLogger:
    """Logger for the JD Optimization application with enhanced caching support"""
    
    def __init__(self, username="Anonymous"):
        """Initialize the logger with session data"""
        self.session_id = str(uuid.uuid4())
        self.username = username
        self.logs_dir = os.path.join(os.getcwd(), "logs")
        
        # Create logs directory if it doesn't exist
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        
        # Initialize the current state
        self.current_state = {
            "session_id": self.session_id,
            "username": self.username,
            "session_start_time": datetime.datetime.now().isoformat(),
            "selected_file": "",
            "original_content": "",
            "enhanced_versions": [],
            "feedback_history": [],
            "final_enhanced_version": "",
            "actions": [],
            "downloads": [],
            "cache": {}  # Add a cache for storing generated content
        }
        
        # Save the initial state
        self._save_state()
    
    def _save_state(self):
        """Save the current state to a file"""
        # Create a filename using the session ID
        filename = f"jdoptim_session_{self.session_id}.json"
        filepath = os.path.join(self.logs_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.current_state, f, indent=2)
        except Exception as e:
            print(f"Error saving state: {e}")
    
    @classmethod
    def load_session(cls, session_id):
        """
        Load an existing session by ID
        
        Args:
            session_id (str): The session ID to load
            
        Returns:
            JDOptimLogger: A logger instance with the loaded state, or None if not found
        """
        logs_dir = os.path.join(os.getcwd(), "logs")
        filename = f"jdoptim_session_{session_id}.json"
        filepath = os.path.join(logs_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Session file not found: {filepath}")
            return None
        
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Create a new logger instance
            logger = cls(state.get("username", "Anonymous"))
            logger.session_id = session_id
            logger.current_state = state
            
            # Ensure cache structure exists
            if "cache" not in logger.current_state:
                logger.current_state["cache"] = {}
                logger._save_state()
            
            return logger
        except Exception as e:
            print(f"Error loading session: {e}")
            return None
    
    @classmethod
    def list_sessions(cls):
        """
        List all available sessions
        
        Returns:
            list: List of session information dictionaries
        """
        logs_dir = os.path.join(os.getcwd(), "logs")
        if not os.path.exists(logs_dir):
            return []
        
        sessions = []
        for filename in os.listdir(logs_dir):
            if filename.startswith("jdoptim_session_") and filename.endswith(".json"):
                session_id = filename.replace("jdoptim_session_", "").replace(".json", "")
                filepath = os.path.join(logs_dir, filename)
                
                try:
                    with open(filepath, 'r') as f:
                        state = json.load(f)
                    
                    sessions.append({
                        "session_id": session_id,
                        "username": state.get("username", "Anonymous"),
                        "start_time": state.get("session_start_time", "Unknown"),
                        "file": state.get("selected_file", "None")
                    })
                except Exception as e:
                    print(f"Error reading session file {filename}: {e}")
        
        # Sort by start time, most recent first
        sessions.sort(key=lambda s: s.get("start_time", ""), reverse=True)
        return sessions
    
    def log_file_selection(self, file_name, content):
        """
        Log file selection
        
        Args:
            file_name (str): Name of the selected file
            content (str): Content of the selected file
        """
        # Update current state
        self.current_state["selected_file"] = file_name
        self.current_state["original_content"] = content
        
        # Add action
        self.current_state["actions"].append({
            "action": "file_selection",
            "file_name": file_name,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Save updated state
        self._save_state()
    
    def log_versions_generated(self, versions):
        """
        Log generated versions
        
        Args:
            versions (list): List of enhanced versions
        """
        # Update current state
        self.current_state["enhanced_versions"] = versions
        
        # Add action
        self.current_state["actions"].append({
            "action": "versions_generated",
            "timestamp": datetime.datetime.now().isoformat(),
            "count": len(versions)
        })
        
        # If we have a unique ID for the current job description, cache the versions
        jd_id = self._get_current_jd_id()
        if jd_id and versions:
            self._cache_versions(jd_id, versions)
        
        # Save updated state
        self._save_state()
    
    def log_version_selection(self, version_index):
        """
        Log version selection
        
        Args:
            version_index (int): Index of the selected version
        """
        # Add action
        self.current_state["actions"].append({
            "action": "version_selection",
            "version_index": version_index,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Save updated state
        self._save_state()
    
    def log_enhanced_version(self, enhanced_content, is_final=False):
        """
        Log enhanced version
        
        Args:
            enhanced_content (str): Enhanced version content
            is_final (bool, optional): Whether this is the final version. Defaults to False.
        """
        # Update current state
        if is_final:
            self.current_state["final_enhanced_version"] = enhanced_content
        
        # Add action
        self.current_state["actions"].append({
            "action": "enhanced_version",
            "is_final": is_final,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # If we have a unique ID for the current job description and this is the final version,
        # cache the final version
        if is_final:
            jd_id = self._get_current_jd_id()
            if jd_id and enhanced_content:
                # Get the base version index from the most recent version selection action
                version_index = 0
                for action in reversed(self.current_state["actions"]):
                    if action["action"] == "version_selection":
                        version_index = action["version_index"]
                        break
                
                self._cache_final_version(jd_id, version_index, enhanced_content)
        
        # Save updated state
        self._save_state()
    
    def log_feedback(self, feedback, feedback_type="General Feedback"):
        """
        Log feedback
        
        Args:
            feedback (str): Feedback content
            feedback_type (str, optional): Type of feedback. Defaults to "General Feedback".
        """
        # Create feedback object
        feedback_obj = {
            "feedback": feedback,
            "type": feedback_type,
            "role": self.username,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Add to feedback history
        self.current_state["feedback_history"].append(feedback_obj)
        
        # Add action
        self.current_state["actions"].append({
            "action": "feedback",
            "timestamp": datetime.datetime.now().isoformat(),
            "index": len(self.current_state["feedback_history"]) - 1,
            "type": feedback_type
        })
        
        # Save updated state
        self._save_state()
    
    def log_download(self, file_type, file_name):
        """
        Log file download
        
        Args:
            file_type (str): Type of file (txt, docx, etc.)
            file_name (str): Name of the downloaded file
        """
        # Add to downloads list
        self.current_state["downloads"].append({
            "file_type": file_type,
            "file_name": file_name,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Add action
        self.current_state["actions"].append({
            "action": "download",
            "file_type": file_type,
            "file_name": file_name,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Save updated state
        self._save_state()
    
    def _get_current_jd_id(self):
        """
        Get a unique ID for the current job description
        
        Returns:
            str: Unique ID for the current job description, or None if not available
        """
        if "jd_unique_id" in self.current_state:
            return self.current_state["jd_unique_id"]
        elif "selected_file" in self.current_state and self.current_state["selected_file"]:
            return f"file_{self.current_state['selected_file']}"
        return None
    
    def _cache_versions(self, jd_id, versions):
        """
        Cache enhanced versions for a job description
        
        Args:
            jd_id (str): Unique ID for the job description
            versions (list): List of enhanced versions
        """
        # Initialize cache structure if it doesn't exist
        if "cache" not in self.current_state:
            self.current_state["cache"] = {}
        
        # Initialize job description entry if it doesn't exist
        if jd_id not in self.current_state["cache"]:
            self.current_state["cache"][jd_id] = {}
        
        # Store enhanced versions
        self.current_state["cache"][jd_id]["enhanced_versions"] = versions
        self.current_state["cache"][jd_id]["timestamp"] = datetime.datetime.now().isoformat()
    
    def _cache_final_version(self, jd_id, version_index, final_version):
        """
        Cache final version for a job description
        
        Args:
            jd_id (str): Unique ID for the job description
            version_index (int): Index of the selected version
            final_version (str): Final enhanced version
        """
        # Initialize cache structure if it doesn't exist
        if "cache" not in self.current_state:
            self.current_state["cache"] = {}
        
        # Initialize job description entry if it doesn't exist
        if jd_id not in self.current_state["cache"]:
            self.current_state["cache"][jd_id] = {}
        
        # Initialize final versions dictionary if it doesn't exist
        if "final_versions" not in self.current_state["cache"][jd_id]:
            self.current_state["cache"][jd_id]["final_versions"] = {}
        
        # Store final version
        self.current_state["cache"][jd_id]["final_versions"][str(version_index)] = final_version
        self.current_state["cache"][jd_id]["final_timestamp"] = datetime.datetime.now().isoformat()
    
    def get_cached_versions(self, jd_id):
        """
        Get cached enhanced versions for a job description
        
        Args:
            jd_id (str): Unique ID for the job description
            
        Returns:
            list or None: List of cached versions if available, None otherwise
        """
        # Check if cache exists
        if "cache" not in self.current_state:
            return None
        
        # Check if job description ID is in cache
        if jd_id in self.current_state["cache"] and "enhanced_versions" in self.current_state["cache"][jd_id]:
            return self.current_state["cache"][jd_id]["enhanced_versions"]
        
        return None
    
    def get_cached_final_version(self, jd_id, version_index):
        """
        Get cached final version for a job description
        
        Args:
            jd_id (str): Unique ID for the job description
            version_index (int): Index of the selected version
            
        Returns:
            str or None: Cached final version if available, None otherwise
        """
        # Check if cache exists
        if "cache" not in self.current_state:
            return None
        
        # Check if job description ID is in cache
        if jd_id in self.current_state["cache"] and "final_versions" in self.current_state["cache"][jd_id]:
            final_versions = self.current_state["cache"][jd_id]["final_versions"]
            
            # Check if we have a final version for this base version
            if str(version_index) in final_versions:
                return final_versions[str(version_index)]
        
        return None
    
    def get_all_feedback(self):
        """
        Get all feedback entries with detailed information
        
        Returns:
            list: List of feedback entries with additional metadata
        """
        feedback_data = []
        
        # Process feedback history
        for i, feedback in enumerate(self.current_state.get("feedback_history", [])):
            # Get basic feedback info
            feedback_text = feedback.get("feedback", "") if isinstance(feedback, dict) else feedback
            feedback_type = feedback.get("type", "General Feedback") if isinstance(feedback, dict) else "General Feedback"
            feedback_role = feedback.get("role", "Unknown") if isinstance(feedback, dict) else "Unknown"
            
            # Find the timestamp from actions
            timestamp = feedback.get("timestamp", "")
            if not timestamp:
                for action in self.current_state.get("actions", []):
                    if action.get("action") == "feedback" and action.get("index", -1) == i:
                        timestamp = action.get("timestamp", "")
                        break
            
            # Format timestamp
            formatted_time = "Unknown"
            if timestamp:
                try:
                    dt = datetime.datetime.fromisoformat(timestamp)
                    formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    formatted_time = str(timestamp)
            
            # Find which job description this feedback was for
            job_desc = self.current_state.get("selected_file", "Unknown JD")
            
            # Add to feedback data
            feedback_data.append({
                "ID": i + 1,
                "Time": formatted_time,
                "Type": feedback_type,
                "Role": feedback_role,
                "Job Description": job_desc,
                "Feedback": feedback_text
            })
        
        return feedback_data