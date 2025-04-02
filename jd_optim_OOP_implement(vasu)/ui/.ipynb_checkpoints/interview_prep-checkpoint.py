import streamlit as st
import pandas as pd
from ui.common import display_section_header, display_subsection_header, display_success_message, display_info_message

def render_interview_prep_page(services):
    """
    Render the interview preparation page with integration to job description
    
    Args:
        services (dict): Dictionary of shared services
    """
    # Unpack services
    logger = services.get('logger')
    analyzer = services.get('analyzer')
    agent = services.get('agent')
    state_manager = services.get('state_manager')
    
    display_section_header("🎯 Interview Preparation")
    
    # First, check if we have an active JD in our repository
    jd_repository = state_manager.get('jd_repository', {})
    jd_content, jd_source_name, jd_unique_id = state_manager.get_jd_content()
    
    if not jd_content:
        # No active JD, notify user
        st.warning("No active job description found.")
        st.info("Please select a job description in the JD Optimization tab first.")
        
        # Show button to navigate to JD Optimization
        if st.button("Go to JD Optimization", key="goto_jd_opt"):
            state_manager.set('active_tab', "JD Optimization")
            st.rerun()
        
        # Show limited functionality
        display_placeholder_content()
    else:
        # Load or generate interview questions based on JD
        st.success(f"Creating interview questions for: {jd_source_name}")
        
        # Check if we already have interview questions for this JD
        interview_data = state_manager.get('interview_data', {})
        if jd_unique_id in interview_data:
            questions = interview_data[jd_unique_id]
            st.info("Using previously generated interview questions")
        else:
            # Generate new questions
            with st.spinner("Generating interview questions..."):
                questions = generate_interview_questions(jd_content, jd_source_name)
                
                # Store in state manager
                interview_data[jd_unique_id] = questions
                state_manager.set('interview_data', interview_data)
        
        # Display generated content
        render_interview_questions(questions, services)

def generate_interview_questions(jd_content, jd_name):
    """
    Generate interview questions based on job description
    
    Args:
        jd_content (str): Job description content
        jd_name (str): Job description name
        
    Returns:
        dict: Generated interview content
    """
    # Extract key skills and requirements
    skills = extract_skills(jd_content)
    
    # For demo purposes, generate standard questions with some customization
    technical_questions = [
        f"Based on the requirement for {skills[0]}, can you explain your experience with this technology?",
        f"How would you implement {skills[1]} in a real-world scenario?",
        "Describe a challenging technical problem you solved recently.",
        f"How do you stay updated with the latest developments in {skills[0]}?",
        "What development methodologies are you familiar with?"
    ]
    
    behavioral_questions = [
        "Tell me about a time when you had to work under a tight deadline.",
        "Describe a situation where you had to collaborate with a difficult team member.",
        "How do you prioritize tasks when working on multiple projects?",
        "Tell me about a time when you received constructive feedback and how you responded.",
        "Describe a project where you demonstrated leadership skills."
    ]
    
    situational_questions = [
        f"How would you approach learning {skills[2]} if it were required for this role?",
        "If a project is falling behind schedule, what steps would you take?",
        "How would you handle a disagreement with a team member about a technical approach?",
        "Describe how you would explain a complex technical concept to a non-technical stakeholder.",
        "If you identified a process that could be improved, how would you approach implementing the change?"
    ]
    
    return {
        'technical_questions': technical_questions,
        'behavioral_questions': behavioral_questions,
        'situational_questions': situational_questions,
        'skills': skills
    }

def extract_skills(jd_content):
    """
    Extract key skills from job description
    
    Args:
        jd_content (str): Job description content
        
    Returns:
        list: Key skills mentioned in JD
    """
    # In a real implementation, you would use NLP to extract skills
    # For demo purposes, use a simple keyword-based approach
    common_skills = [
        'Python', 'Java', 'JavaScript', 'C++', 'SQL', 'AWS', 'Azure', 
        'Machine Learning', 'Data Analysis', 'DevOps', 'React', 'Angular',
        'Cloud Computing', 'Docker', 'Kubernetes', 'Agile', 'CI/CD'
    ]
    
    found_skills = []
    jd_lower = jd_content.lower()
    
    for skill in common_skills:
        if skill.lower() in jd_lower:
            found_skills.append(skill)
    
    # Ensure we have at least 3 skills
    while len(found_skills) < 3:
        found_skills.append(common_skills[len(found_skills)])
    
    return found_skills

def render_interview_questions(questions, services):
    """
    Render the interview questions in a structured format
    
    Args:
        questions (dict): Dictionary of questions
        services (dict): Dictionary of shared services
    """
    # Create tabs for different question types
    tabs = st.tabs(["Technical Questions", "Behavioral Questions", "Situational Questions", "Evaluation Criteria"])
    
    # Technical Questions Tab
    with tabs[0]:
        display_subsection_header("Technical Interview Questions")
        technical_questions = questions.get('technical_questions', [])
        
        if technical_questions:
            for i, question in enumerate(technical_questions, 1):
                with st.expander(f"Question {i}: {question[:50]}...", expanded=i == 1):
                    st.write(question)
                    
                    # Add note-taking area
                    st.text_area(f"Notes for Question {i}", key=f"tech_notes_{i}", height=100)
                    
                    # Add rating
                    st.select_slider(
                        f"Rate Answer for Question {i}", 
                        options=["Poor", "Below Average", "Average", "Good", "Excellent"],
                        value="Average",
                        key=f"tech_rating_{i}"
                    )
        else:
            st.info("No technical questions available.")
            
        # Add custom question option
        st.markdown("### Add Custom Technical Question")
        custom_tech = st.text_area("Enter your custom technical question:", key="custom_tech")
        if st.button("Add Question", key="add_tech"):
            if custom_tech:
                technical_questions.append(custom_tech)
                questions['technical_questions'] = technical_questions
                st.rerun()
    
    # Behavioral Questions Tab
    with tabs[1]:
        display_subsection_header("Behavioral Interview Questions")
        behavioral_questions = questions.get('behavioral_questions', [])
        
        if behavioral_questions:
            for i, question in enumerate(behavioral_questions, 1):
                with st.expander(f"Question {i}: {question[:50]}...", expanded=i == 1):
                    st.write(question)
                    
                    # Add note-taking area
                    st.text_area(f"Notes for Question {i}", key=f"behav_notes_{i}", height=100)
                    
                    # Add rating
                    st.select_slider(
                        f"Rate Answer for Question {i}", 
                        options=["Poor", "Below Average", "Average", "Good", "Excellent"],
                        value="Average",
                        key=f"behav_rating_{i}"
                    )
        else:
            st.info("No behavioral questions available.")
            
        # Add custom question option
        st.markdown("### Add Custom Behavioral Question")
        custom_behav = st.text_area("Enter your custom behavioral question:", key="custom_behav")
        if st.button("Add Question", key="add_behav"):
            if custom_behav:
                behavioral_questions.append(custom_behav)
                questions['behavioral_questions'] = behavioral_questions
                st.rerun()
    
    # Situational Questions Tab
    with tabs[2]:
        display_subsection_header("Situational Interview Questions")
        situational_questions = questions.get('situational_questions', [])
        
        if situational_questions:
            for i, question in enumerate(situational_questions, 1):
                with st.expander(f"Question {i}: {question[:50]}...", expanded=i == 1):
                    st.write(question)
                    
                    # Add note-taking area
                    st.text_area(f"Notes for Question {i}", key=f"sit_notes_{i}", height=100)
                    
                    # Add rating
                    st.select_slider(
                        f"Rate Answer for Question {i}", 
                        options=["Poor", "Below Average", "Average", "Good", "Excellent"],
                        value="Average",
                        key=f"sit_rating_{i}"
                    )
        else:
            st.info("No situational questions available.")
            
        # Add custom question option
        st.markdown("### Add Custom Situational Question")
        custom_sit = st.text_area("Enter your custom situational question:", key="custom_sit")
        if st.button("Add Question", key="add_sit"):
            if custom_sit:
                situational_questions.append(custom_sit)
                questions['situational_questions'] = situational_questions
                st.rerun()
    
    # Evaluation Criteria Tab
    with tabs[3]:
        display_subsection_header("Candidate Evaluation Criteria")
        
        # Create evaluation criteria based on skills
        skills = questions.get('skills', [])
        
        st.markdown("### Technical Skills Assessment")
        
        # Create dataframe for skills evaluation
        criteria_data = []
        for skill in skills:
            criteria_data.append({
                "Criterion": f"{skill} Proficiency",
                "Weight": "High",
                "Description": f"Evaluates candidate's knowledge and practical experience with {skill}"
            })
        
        # Add general criteria
        general_criteria = [
            {
                "Criterion": "Problem-Solving Ability",
                "Weight": "High",
                "Description": "Assesses how effectively the candidate analyzes problems and implements solutions"
            },
            {
                "Criterion": "Communication Skills",
                "Weight": "Medium",
                "Description": "Evaluates clarity of communication and ability to explain technical concepts"
            },
            {
                "Criterion": "Team Collaboration",
                "Weight": "Medium",
                "Description": "Assesses ability to work effectively with team members and stakeholders"
            },
            {
                "Criterion": "Cultural Fit",
                "Weight": "Medium",
                "Description": "Evaluates alignment with company values and team dynamics"
            },
            {
                "Criterion": "Learning Agility",
                "Weight": "High",
                "Description": "Assesses ability to quickly learn new technologies and adapt to changes"
            }
        ]
        
        # Combine all criteria
        criteria_data.extend(general_criteria)
        
        # Display as a table
        criteria_df = pd.DataFrame(criteria_data)
        st.dataframe(criteria_df, use_container_width=True, hide_index=True)
        
        # Scoring guide
        st.markdown("### Scoring Guide")
        scoring_data = [
            {"Score": "Poor (1)", "Description": "Does not meet minimum expectations for the role"},
            {"Score": "Below Average (2)", "Description": "Meets some expectations but has significant gaps"},
            {"Score": "Average (3)", "Description": "Meets basic expectations for the role"},
            {"Score": "Good (4)", "Description": "Exceeds expectations in some areas"},
            {"Score": "Excellent (5)", "Description": "Exceeds expectations in most or all areas"}
        ]
        
        scoring_df = pd.DataFrame(scoring_data)
        st.dataframe(scoring_df, use_container_width=True, hide_index=True)
    
    # Add download buttons for the interview packet
    st.markdown("---")
    st.markdown("### Interview Packet")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📄 Generate Interview Packet (PDF)", key="gen_pdf"):
            # In a real implementation, this would generate a PDF
            display_success_message("Interview packet generated! (Demo - no actual PDF created)")
    
    with col2:
        if st.button("📊 Generate Scorecard Template (Excel)", key="gen_excel"):
            # In a real implementation, this would generate an Excel file
            display_success_message("Scorecard template generated! (Demo - no actual Excel file created)")

def display_placeholder_content():
    """Display placeholder content when no JD is selected"""
    st.markdown("""
    <div style="text-align: center; padding: 40px; background-color: #f8f9fa; border-radius: 10px; margin: 20px 0;">
        <img src="https://img.icons8.com/cotton/100/000000/interview.png" alt="Interview Prep" width="64" height="64">
        <h2 style="margin-top: 20px; color: #1e3a8a;">Interview Preparation Tool</h2>
        <p style="color: #6b7280; max-width: 600px; margin: 0 auto; padding: 10px 0;">
            Create structured interview questions, evaluation criteria, and candidate 
            scoring templates based on your job descriptions.
        </p>
        <p style="color: #6b7280; max-width: 600px; margin: 10px auto;">
            Please select a job description in the JD Optimization tab to get started.
        </p>
    </div>
    """, unsafe_allow_html=True)