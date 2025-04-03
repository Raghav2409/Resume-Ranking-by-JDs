import json
import re
import streamlit as st

class JobDescriptionAgent:
    """Agent for enhancing job descriptions (offline version for testing)"""
    def __init__(self, model_id, max_tokens=10000, temperature=0.7):
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # For offline testing - no actual API connection
        self.client = None

    def _invoke_bedrock_model(self, prompt):
        """Simulate model invocation for offline testing"""
        # This function simulates a response without calling the actual API
        sample_response = {
            "content": [
                {"text": "This is a simulated response for testing purposes."}
            ]
        }
        return sample_response
            
    def generate_initial_descriptions(self, job_description):
        """Generate detailed and structured job descriptions based on the given job description."""
        # Create offline versions for testing
        return [
            self._generate_version_1(job_description),
            self._generate_version_2(job_description),
            self._generate_version_3(job_description)
        ]
    
    def _generate_version_1(self, job_description):
        """Generate version 1 with focus on technical skills"""
        return f"""VERSION 1:

Role Overview:
This role is responsible for delivering high-quality solutions by applying technical expertise and industry best practices. The position requires collaboration with cross-functional teams to achieve project objectives while ensuring code quality and performance.

Key Responsibilities:
• Design, develop, and maintain software applications according to requirements
• Collaborate with product managers and stakeholders to refine specifications
• Write clean, efficient, and well-documented code
• Conduct code reviews and implement best practices
• Troubleshoot and resolve complex technical issues
• Participate in agile development processes and team meetings

Required Skills:
• Proficiency in key programming languages and frameworks
• Strong knowledge of software development principles
• Excellent problem-solving and analytical abilities
• Experience with version control systems (Git)
• Understanding of database concepts and SQL/NoSQL technologies

Preferred Skills:
• Experience with cloud platforms (AWS, Azure, or GCP)
• Knowledge of containerization (Docker, Kubernetes)
• Familiarity with CI/CD pipelines
• Understanding of microservices architecture
• Experience with test-driven development

Required Experience:
• 3+ years of experience in software development
• History of successful project delivery
• Experience working in agile development environments
• Track record of writing maintainable, scalable code

Preferred Experience:
• Experience in a similar industry domain
• History of mentoring junior developers
• Participation in open-source projects
• Experience with system design and architecture

Tools & Technologies:
• Modern development frameworks and libraries
• Database systems (SQL and NoSQL)
• Version control systems
• Code quality and testing tools
• Continuous integration and deployment tools

Work Environment & Expectations:
• Collaborative team environment
• Agile development methodology
• Focus on continuous learning and improvement
• Regular code reviews and knowledge sharing
• Balance of independent work and team collaboration

Enhanced from original: {job_description[:100]}..."""
    
    def _generate_version_2(self, job_description):
        """Generate version 2 with focus on soft skills and culture"""
        return f"""VERSION 2:

Role Overview:
This role is pivotal in our organization's success, requiring a blend of technical expertise and strong collaboration skills. The ideal candidate will drive innovation while ensuring quality deliverables that meet business objectives and user needs.

Key Responsibilities:
• Develop and implement solutions that address complex business challenges
• Work closely with business analysts to understand and refine requirements
• Ensure code quality through proper testing and validation procedures
• Mentor team members and share knowledge to elevate team capabilities
• Identify and propose improvements to existing systems and processes
• Participate in planning sessions and contribute to technical decision-making

Required Skills:
• Strong technical proficiency in relevant programming languages
• Excellent communication and interpersonal abilities
• Problem-solving mindset with attention to detail
• Ability to work independently and as part of a team
• Adaptability and willingness to learn new technologies

Preferred Skills:
• Experience with Agile/Scrum methodologies
• Knowledge of UX/UI design principles
• Project management skills
• Technical writing and documentation abilities
• Critical thinking and solution architecture skills

Required Experience:
• Minimum 3 years in a similar role
• Experience in full software development lifecycle
• History of successful collaboration with cross-functional teams
• Background in delivering projects on time and within scope

Preferred Experience:
• Experience in our specific industry vertical
• Remote or distributed team collaboration
• Client-facing experience
• History of leading technical initiatives

Tools & Technologies:
• Industry-standard development environments
• Collaboration and project management tools
• Documentation systems
• Testing and quality assurance tools
• Communication and knowledge sharing platforms

Work Environment & Expectations:
• Supportive team culture focused on growth
• Emphasis on work-life balance
• Regular opportunities for professional development
• Environment that values diversity of thought and approach
• Results-oriented with flexible working arrangements

Enhanced from original: {job_description[:100]}..."""
    
    def _generate_version_3(self, job_description):
        """Generate version 3 with focus on business impact"""
        return f"""VERSION 3:

Role Overview:
This role is essential for driving business value through technical excellence. The position requires a strategic mindset to develop solutions that enhance operational efficiency, customer satisfaction, and competitive advantage while maintaining high standards of quality and security.

Key Responsibilities:
• Create innovative solutions that align with business objectives and user needs
• Analyze requirements and translate them into technical specifications
• Develop scalable and maintainable code following established standards
• Optimize application performance and resource utilization
• Identify technical debt and implement strategies to address it
• Collaborate with stakeholders to ensure solutions meet business needs

Required Skills:
• Strong programming abilities with relevant languages and frameworks
• Excellent analytical and logical reasoning capabilities
• Ability to translate business requirements into technical solutions
• Understanding of software architecture principles
• Knowledge of security best practices and implementation

Preferred Skills:
• Familiarity with data analytics and business intelligence
• Understanding of industry regulations and compliance requirements
• Experience with performance optimization techniques
• Knowledge of accessibility standards
• Capacity for technical leadership and decision-making

Required Experience:
• 3+ years of professional development experience
• History of delivering business-critical applications
• Experience working with databases and API integrations
• Background in creating scalable, maintainable solutions

Preferred Experience:
• Experience in a similar business domain
• History of improving system performance or reliability
• Background in transitioning legacy systems to modern architectures
• Experience with cost optimization in development

Tools & Technologies:
• Enterprise-grade development tools
• Business intelligence and reporting systems
• Monitoring and logging frameworks
• Performance testing tools
• Security validation and compliance tools

Work Environment & Expectations:
• Business-focused development culture
• Emphasis on measurable outcomes and impact
• Regular interaction with business stakeholders
• Environment that balances innovation with reliability
• Commitment to continuous improvement and excellence

Enhanced from original: {job_description[:100]}..."""
        
    def generate_final_description(self, selected_description, feedback_history):
        """
        Generate enhanced description incorporating feedback history
        
        Args:
            selected_description (str): The base description to enhance
            feedback_history (list): List of previous feedback items
        """
        # If no feedback, just return the selected description
        if not feedback_history:
            return selected_description
            
        # Extract feedback text
        feedback_text = []
        for item in feedback_history:
            if isinstance(item, dict):
                feedback_text.append(item.get("feedback", ""))
            else:
                feedback_text.append(str(item))
        
        # Create a simple enhanced version with feedback incorporated
        feedback_summary = "\n".join([f"• {f}" for f in feedback_text if f])
        
        enhanced = f"""
ENHANCED JOB DESCRIPTION

{selected_description}

INCORPORATED FEEDBACK:
{feedback_summary}

This enhanced job description addresses all feedback while maintaining professional quality and clarity.
        """
        
        return enhanced