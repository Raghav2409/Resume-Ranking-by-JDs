import json
import re
import boto3
import streamlit as st

class JobDescriptionAgent:
    """Agent for enhancing job descriptions using AWS Bedrock Claude"""
    def __init__(self, model_id, max_tokens=5000, temperature=0.7):
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Initialize AWS client for Bedrock
        try:
            # Use Streamlit secrets for credentials in production
            self.client = boto3.client(
                service_name='bedrock-runtime',
                aws_access_key_id=st.secrets["aws"]["access_key"],
                aws_secret_access_key=st.secrets["aws"]["secret_key"],
                region_name=st.secrets["aws"]["region"],
            )
        except Exception as e:
            print(f"Error initializing AWS Bedrock client: {e}")
            self.client = None

    def _invoke_bedrock_model(self, prompt):
        """Private method to invoke the Bedrock model with a prompt"""
        if not self.client:
            return None
            
        try:
            native_request = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
            
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(native_request),
                contentType="application/json",
            )
            
            response_body = response['body'].read().decode("utf-8")
            return json.loads(response_body)
        except Exception as e:
            print(f"Error invoking Bedrock model: {e}")
            return None
            
    def generate_initial_descriptions(self, job_description):
        """Generate detailed and structured job descriptions based on the given job description."""
        # If client is not initialized properly, return dummy versions
        if not self.client:
            return [
                f"Enhanced Version 1 (Example):\n\nOverview: This role is responsible for...\n\nKey Responsibilities:\n- Responsibility 1\n- Responsibility 2\n\nRequired Skills:\n- Skill 1\n- Skill 2",
                f"Enhanced Version 2 (Example):\n\nOverview: This position focuses on...\n\nKey Responsibilities:\n- Responsibility A\n- Responsibility B\n\nRequired Skills:\n- Skill A\n- Skill B",
                f"Enhanced Version 3 (Example):\n\nOverview: A key position that...\n\nKey Responsibilities:\n- Primary task 1\n- Primary task 2\n\nRequired Skills:\n- Critical skill 1\n- Critical skill 2"
            ]
        
        prompt = (
            "You are a job description specialist. Your task is to refine and expand upon the provided job description, "
            "creating three distinct versions that are structured, detailed, and aligned with industry best practices.\n\n"
            
            "### Guidelines:\n"
            "- Do NOT make assumptions or introduce inaccuracies.\n"
            "- Avoid using specific job titles; refer to the position as **'this role'** throughout.\n"
            "- Each version should be unique, emphasizing different aspects of the role.\n"
            "- Ensure clarity, conciseness, and engagement in the descriptions.\n\n"
            
            "### Structure for Each Job Description:\n"
            "**1. Role Overview:** A compelling and detailed explanation of this role's significance.\n"
            "**2. Key Responsibilities:** Bullet points outlining core duties, including specifics where applicable.\n"
            "**3. Required Skills:** Essential technical and soft skills, with explanations of their importance.\n"
            "**4. Preferred Skills:** Additional skills that would be advantageous, with context on their relevance.\n"
            "**5. Required Experience:** The necessary experience levels, with examples of relevant past roles.\n"
            "**6. Preferred Experience:** Additional experience that would enhance performance in this role.\n"
            "**7. Tools & Technologies:** Key tools, software, and technologies required for this role.\n"
            "**8. Work Environment & Expectations:** Details on work conditions, methodologies, or collaboration requirements.\n\n"
        
            "Ensure each job description expands on the provided details, enhancing clarity and depth while maintaining industry relevance.\n\n"
            "### Required Format:\n"
            "Present your response exactly as follows:\n\n"
            
            "VERSION 1:\n"
            "[Complete first job description with all sections]\n\n"
            
            "VERSION 2:\n"
            "[Complete second job description with all sections]\n\n"
            
            "VERSION 3:\n"
            "[Complete third job description with all sections]\n\n"
            
            f"### Original Job Description:\n{job_description}\n"
        )

        model_response = self._invoke_bedrock_model(prompt)
        
        try:
            if model_response and "content" in model_response and isinstance(model_response["content"], list):
                full_text = model_response["content"][0]["text"].strip()
                
                # More robust splitting pattern
                parts = re.split(r'VERSION \d+:', full_text)
                if len(parts) >= 4:  # The first part is empty or intro text
                    descriptions = [part.strip() for part in parts[1:4]]
                    return descriptions
                else:
                    # Fallback parsing method
                    descriptions = []
                    version_pattern = re.compile(r'VERSION (\d+):(.*?)(?=VERSION \d+:|$)', re.DOTALL)
                    matches = version_pattern.findall(full_text)
                    for _, content in matches[:3]:
                        descriptions.append(content.strip())
                    
                    if len(descriptions) == 3:
                        return descriptions
        except Exception as e:
            print(f"Error parsing generated descriptions: {e}")
        
        # If we failed to parse properly or encountered an error, generate simpler versions
        return [
            f"Enhanced Version 1 of the job description:\n{job_description}",
            f"Enhanced Version 2 of the job description:\n{job_description}",
            f"Enhanced Version 3 of the job description:\n{job_description}"
        ]

    def generate_final_description(self, selected_description, feedback_history):
        """
        Generate enhanced description incorporating feedback history
        
        Args:
            selected_description (str): The base description to enhance
            feedback_history (list): List of previous feedback items
        """
        # If client is not initialized properly, return the selected description
        if not self.client:
            return selected_description + "\n\n[Note: This would normally be enhanced based on your feedback, but the AI service connection is currently unavailable.]"
            
        # Construct prompt with feedback history
        feedback_context = ""
        for i, feedback_item in enumerate(feedback_history[:-1]):
            if isinstance(feedback_item, dict):
                feedback_type = feedback_item.get("type", "General Feedback")
                feedback_text = feedback_item.get("feedback", "")
                feedback_context += f"Previous Feedback {i+1} ({feedback_type}): {feedback_text}\n\n"
            else:
                feedback_context += f"Previous Feedback {i+1}: {feedback_item}\n\n"
        
        # Handle current feedback
        current_feedback = ""
        if feedback_history:
            last_feedback = feedback_history[-1]
            if isinstance(last_feedback, dict):
                feedback_type = last_feedback.get("type", "General Feedback")
                feedback_text = last_feedback.get("feedback", "")
                current_feedback = f"({feedback_type}): {feedback_text}"
            else:
                current_feedback = last_feedback
        
        prompt = (
            "You are an expert in job description refinement. Your task is to enhance the given job description "
            "by incorporating all feedback while maintaining professional quality.\n\n"
            
            f"### Selected Job Description to Enhance:\n{selected_description}\n\n"
        )
        if feedback_context:
            prompt += f"### Previous Feedback Already Incorporated:\n{feedback_context}\n\n"
        
        if current_feedback:
            prompt += f"### New Feedback to Implement:\n{current_feedback}\n\n"
        
        prompt += (
                "### Guidelines:\n"
                "- Implement all feedback while preserving the original core requirements\n"
                "- Maintain clear section structure and professional language\n"
                "- Continue referring to the position as 'this role'\n"
                "- Produce a complete, refined job description ready for immediate use\n\n"
                
                "Return the complete enhanced job description incorporating all feedback."
            )
        
        model_response = self._invoke_bedrock_model(prompt)
        
        try:
            if model_response and "content" in model_response and isinstance(model_response["content"], list):
                return model_response["content"][0]["text"].strip()
        except Exception as e:
            print(f"Error generating final description: {e}")
            
        return selected_description + f"\n\n[Error generating final version: Unable to process feedback]"