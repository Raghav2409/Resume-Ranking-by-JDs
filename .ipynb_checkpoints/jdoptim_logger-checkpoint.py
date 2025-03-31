import os
import json
import datetime
import time
from typing import Dict, List, Any, Optional
import uuid

#Helper Class for log files
class JDOptimLogger:
    """
    Logger class for tracking state changes in the Job Description Optimizer application.
    Maintains a history of user interactions, feedback, and version changes in .Json format.
    """
    
    def __init__(self, log_dir: str = "logs", username: str = "Anonymous"):
        """
        Initialize the logger
        
        Args:
            log_dir: Directory to store log files
            username: Username for tracking who made changes
        """
        self.username = username
        self.log_dir = log_dir
        self.session_id = str(uuid.uuid4())
        self.session_start_time = datetime.datetime.now().isoformat()
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create log file with session ID
        self.log_file = os.path.join(
            self.log_dir, 
            f"jdoptim_session_{self.session_id}.json"
        )
        
        # Initialize state
        self.current_state = {
            "session_id": self.session_id,
            "username": self.username,
            "session_start_time": self.session_start_time,
            "selected_file": None,
            "original_jd": None,
            "enhancioed_versns": [],
            "selected_version": None,
            "feedback_history": [],
            "current_enhanced_version": None,
            "final_version": None,
            "actions": []
        }
        
        # Save initial state
        self._save_state()
    
    def _save_state(self) -> None:
        """Save the current state to the log file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.current_state, f, indent=2)

    def log_file_selection(self, file_name: str, content: str) -> None:
        """
        Log when a user selects a file
        
        Args:
            file_name: Name of the selected file
            content: Content of the job description
        """
        # Only log if the file has changed or is not set
        if self.current_state["selected_file"] != file_name:
            self.current_state["selected_file"] = file_name
            self.current_state["original_jd"] = content
            # Reset version-related state when a new file is selected
            self.current_state["enhanced_versions"] = []
            self.current_state["selected_version"] = None
            self.current_state["feedback_history"] = []
            self.current_state["current_enhanced_version"] = None
            self.current_state["final_version"] = None
            
            self.current_state["actions"].append({
                "action": "file_selected",
                "file_name": file_name,
                "username": self.username,
                "timestamp": datetime.datetime.now().isoformat()
            })
            self._save_state()
            return True
        return False

    def log_versions_generated(self, versions: List[str]) -> None:
        """
        Log when enhanced versions are generated
            
        Args:
            versions: List of enhanced job description versions
        """
        # Only log if versions have changed or are not set
        if not self.current_state["enhanced_versions"] or self.current_state["enhanced_versions"] != versions:
            self.current_state["enhanced_versions"] = versions
            self.current_state["actions"].append({
                "action": "versions_generated",
                "version_count": len(versions),
                "username": self.username,
                "timestamp": datetime.datetime.now().isoformat()
            })
            self._save_state()
            return True
        return False
        
    def log_version_selection(self, version_index: int) -> None:
        """
        Log when a user selects a version for further enhancement
        
        Args:
            version_index: Index of the selected version (0-based)
        """
        # Only log if version selection has changed
        if self.current_state["selected_version"] != version_index:
            self.current_state["selected_version"] = version_index
            self.current_state["actions"].append({
                "action": "version_selected",
                "version_index": version_index,
                "username": self.username,
                "timestamp": datetime.datetime.now().isoformat()
            })
            self._save_state()
            return True
        return False
    
    def log_feedback(self, feedback):
        """
        Log user feedback for job description enhancement
        
        Args:
            feedback: Feedback text or list of feedback items
        """
        if not feedback:
            return False
        
        # Check if this feedback already exists
        if isinstance(feedback, str):
            # Don't add duplicate feedback
            for existing_feedback in self.current_state["feedback_history"]:
                if isinstance(existing_feedback, str) and existing_feedback == feedback:
                    return False
                elif isinstance(existing_feedback, dict) and existing_feedback.get("feedback") == feedback:
                    return False
            
            # Add new feedback
            self.current_state["feedback_history"].append(feedback)
            
            # Get the index of the feedback
            feedback_index = len(self.current_state["feedback_history"]) - 1
            
            # Use ISO format timestamp for consistency
            timestamp = datetime.datetime.now().isoformat()
            self.current_state["actions"].append({
                "action": "feedback",
                "timestamp": timestamp,
                "feedback": feedback,
                "index": feedback_index,
                "username": self.username
            })
            
            # Save changes
            self._save_state()
            return True
        
        elif isinstance(feedback, list):
            # Add all new feedback items
            added = False
            for item in feedback:
                if self.log_feedback(item):
                    added = True
            return added
            
        return False
            
    def log_enhanced_version(self, enhanced_version: str, is_final: bool = False) -> None:
        """
        Log when an enhanced version is generated after feedback
        
        Args:
            enhanced_version: The enhanced job description
            is_final: Whether this is the final version
        """
        # Check if version has changed
        version_changed = (self.current_state["current_enhanced_version"] != enhanced_version)
        
        if version_changed:
            self.current_state["current_enhanced_version"] = enhanced_version
            
            if is_final:
                self.current_state["final_version"] = enhanced_version
            
            self.current_state["actions"].append({
                "action": "enhanced_version_generated",
                "is_final": is_final,
                "username": self.username,
                "timestamp": datetime.datetime.now().isoformat()
            })
            self._save_state()
            return True
        
        return False
    
    def log_download(self, format_type: str, file_path: str) -> None:
        """
        Log when a user downloads an enhanced job description
        
        Args:
            format_type: The format of the download (txt, docx)
            file_path: Path where the file was saved
        """
        self.current_state["actions"].append({
            "action": "file_downloaded",
            "format": format_type,
            "file_path": file_path,
            "username": self.username,
            "timestamp": datetime.datetime.now().isoformat()
        })
        self._save_state()
    
    def get_session_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of the current session
        
        Returns:
            Dictionary containing session summary
        """
        actions_count = {}
        for action in self.current_state["actions"]:
            action_type = action["action"]
            if action_type in actions_count:
                actions_count[action_type] += 1
            else:
                actions_count[action_type] = 1
        
        session_duration = None
        if self.current_state["actions"]:
            try:
                start_time = datetime.datetime.fromisoformat(self.session_start_time)
                last_action_time = datetime.datetime.fromisoformat(
                    self.current_state["actions"][-1]["timestamp"]
                )
                session_duration = (last_action_time - start_time).total_seconds()
            except:
                session_duration = 0
        
        return {
            "session_id": self.session_id,
            "username": self.username,
            "start_time": self.session_start_time,
            "file_processed": self.current_state["selected_file"],
            "feedback_count": len(self.current_state["feedback_history"]),
            "actions_summary": actions_count,
            "session_duration_seconds": session_duration,
            "has_final_version": self.current_state["final_version"] is not None
        }
    
    @staticmethod
    def load_session(session_id: str, log_dir: str = "logs") -> Optional['JDOptimLogger']:
        """
        Load a previous session by ID
        
        Args:
            session_id: ID of the session to load
            log_dir: Directory containing log files
            
        Returns:
            JDOptimLogger instance or None if session not found
        """
        # Find the file with the session_id in its name
        if not os.path.exists(log_dir):
            return None
            
        matching_files = []
        for file_name in os.listdir(log_dir):
            if file_name.endswith('.json') and session_id in file_name:
                matching_files.append(file_name)
                
        if not matching_files:
            return None
            
        # Use the first matching file
        log_file = os.path.join(log_dir, matching_files[0])
        
        try:
            with open(log_file, 'r') as f:
                state = json.load(f)
                
            # Extract username from state
            username = state.get("username", "Anonymous")
            
            # Create logger with the session ID and username
            logger = JDOptimLogger(log_dir, username)
            
            # Override generated session ID with the loaded one
            logger.session_id = state["session_id"]
            logger.log_file = log_file
            
            # Replace the initial state with the loaded state
            logger.current_state = state
            
            # Make sure username is current
            logger.current_state["username"] = username
            
            return logger
        except Exception as e:
            print(f"Error loading session: {str(e)}")
            return None
    
    @staticmethod
    def list_sessions(log_dir: str = "logs") -> List[Dict[str, Any]]:
        """
        List all available session logs
        
        Args:
            log_dir: Directory containing log files
            
        Returns:
            List of session summaries
        """
        if not os.path.exists(log_dir):
            return []
        
        sessions = []
        for file_name in os.listdir(log_dir):
            if file_name.startswith("jdoptim_session_") and file_name.endswith(".json"):
                try:
                    file_path = os.path.join(log_dir, file_name)
                    with open(file_path, 'r') as f:
                        state = json.load(f)
                    
                    # Extract session ID from state
                    session_id = state.get("session_id", "Unknown")
                    
                    # Extract username from state
                    username = state.get("username", "Anonymous")
                    
                    # Parse session start time to a readable format
                    start_time = state.get("session_start_time", "Unknown")
                    try:
                        if start_time != "Unknown":
                            dt = datetime.datetime.fromisoformat(start_time)
                            start_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        pass
                    
                    # Create a detailed summary
                    sessions.append({
                        "session_id": session_id,
                        "username": username,
                        "start_time": start_time,
                        "file_processed": state.get("selected_file", "Unknown"),
                        "action_count": len(state.get("actions", [])),
                        "has_final_version": state.get("final_version") is not None
                    })
                except Exception as e:
                    print(f"Error reading session file {file_name}: {str(e)}")
        
        # Sort by start time (newest first)
        sessions.sort(key=lambda x: x["start_time"], reverse=True)
        return sessions
        
    @staticmethod
    def cleanup_old_sessions(max_sessions=100, max_days=30, log_dir="logs"):
        """Clean up old session files to prevent excessive accumulation"""
        if not os.path.exists(log_dir):
            return
            
        session_files = []
        for file in os.listdir(log_dir):
            if file.endswith('.json') and file.startswith('jdoptim_session_'):
                file_path = os.path.join(log_dir, file)
                modified_time = os.path.getmtime(file_path)
                session_files.append((file_path, modified_time))
        
        # Sort in chronological order (newest first)
        session_files.sort(key=lambda x: x[1], reverse=True)
        
        # Keep the newest max_sessions files, delete the rest
        if len(session_files) > max_sessions:
            for file_path, _ in session_files[max_sessions:]:
                try:
                    os.remove(file_path)
                except:
                    pass
        
        # Also delete files older than max_days
        cutoff_time = time.time() - (max_days * 24 * 60 * 60)
        for file_path, mod_time in session_files[:max_sessions]:  # Check the kept files too
            if mod_time < cutoff_time:
                try:
                    os.remove(file_path)
                except:
                    pass
              #di      
    def export_session_report(self, export_dir: Optional[str] = None) -> str:
        """
        Export a human-readable report of the session
        
        Args:
            export_dir: Directory to save the report (defaults to log_dir)
            
        Returns:
            Path to the exported report file
        """
        if export_dir is None:
            export_dir = self.log_dir
            
        os.makedirs(export_dir, exist_ok=True)
        
        # Generate timestamp for filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(export_dir, f"jdoptim_report_{self.username}_{timestamp}.txt")
        
        with open(report_file, 'w') as f:
            f.write(f"JD OPTIMIZER SESSION REPORT\n")
            f.write(f"=========================\n\n")
            
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"User: {self.username}\n")
            f.write(f"Start Time: {self.session_start_time}\n")
            f.write(f"File Processed: {self.current_state['selected_file']}\n\n")
            
            # Feedback history
            f.write(f"FEEDBACK HISTORY ({len(self.current_state['feedback_history'])} items):\n")
            for i, feedback_item in enumerate(self.current_state['feedback_history'], 1):
                # Handle both old format (string) and new format (dict)
                if isinstance(feedback_item, str):
                    feedback = feedback_item
                else:
                    feedback = feedback_item.get("feedback", "")
                
                f.write(f"{i}. {feedback}\n")
            f.write("\n")
            
            # Actions timeline
            f.write(f"ACTIONS TIMELINE ({len(self.current_state['actions'])} actions):\n")
            for i, action in enumerate(self.current_state['actions'], 1):
                # Format timestamp
                action_time = "Unknown"
                if "timestamp" in action:
                    try:
                        dt = datetime.datetime.fromisoformat(action['timestamp'])
                        action_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        action_time = action['timestamp']
                
                action_type = action['action']
                
                details = ""
                if action_type == "file_selected":
                    details = f"File: {action.get('file_name', 'Unknown')}"
                elif action_type == "versions_generated":
                    details = f"Generated {action.get('version_count', 0)} versions"
                elif action_type == "version_selected":
                    details = f"Selected version {action.get('version_index', 0) + 1}"
                elif action_type == "feedback":
                    details = f"Feedback: {action.get('feedback', '')[:50]}..."
                elif action_type == "enhanced_version_generated":
                    details = f"{'Final' if action.get('is_final', False) else 'Intermediate'} version"
                elif action_type == "file_downloaded":
                    details = f"Format: {action.get('format', 'Unknown')}"
                
                f.write(f"{i}. [{action_time}] {action_type}: {details}\n")
            
            return report_file