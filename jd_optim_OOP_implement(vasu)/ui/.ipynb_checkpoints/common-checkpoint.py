import streamlit as st

def render_header():
    """Render the application header with logo and title"""
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
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center; align-items: center; height: 60px;">
                <div style="padding: 5px 10px; border-radius: 5px; text-align: center;">
                    <div style="font-weight: bold;">{st.session_state.role}</div>
                    <div style="font-size: 0.8em;">View</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

def render_role_selector():
    """Render the role selector in a compact layout"""
    # Display role selector in a small container with border
    with st.container(border=True):
        # Define available roles
        roles = ["Recruiter", "Hiring Manager", "Candidate", "HR Manager", "Team Lead"]
        
        # Simple one-row layout
        selected_role = st.selectbox(
            "Your Role:",
            options=roles,
            index=roles.index(st.session_state.role) if st.session_state.role in roles else 0,
            help="Select your role in the hiring process"
        )
        
        # Update role if changed
        if selected_role != st.session_state.role:
            st.session_state.role = selected_role
            
            # Update logger if it exists
            if 'logger' in st.session_state:
                st.session_state.logger.username = selected_role
                st.session_state.logger.current_state["username"] = selected_role
                st.session_state.logger._save_state()

def render_tabs():
    """Render the navigation tabs"""
    tabs = ["JD Versions", "Feedback Loop", "Candidate Ranking", "Client Feedback", "Interview Prep"]
    
    # Create tab buttons
    cols = st.columns(len(tabs))
    
    for i, tab in enumerate(tabs):
        with cols[i]:
            is_active = st.session_state.active_tab == tab
            bg_color = "#DBEAFE" if is_active else "#F9FAFB"
            text_color = "#1E40AF" if is_active else "#374151"
            border_bottom = "3px solid #2563EB" if is_active else "1px solid #E5E7EB"
            
            # Create a button that looks like a tab
            if st.button(
                tab,
                key=f"tab_{tab}",
                use_container_width=True,
                type="secondary" if is_active else "secondary",
                help=f"Switch to {tab} tab"
            ):
                switch_tab(tab)

def switch_tab(tab_name):
    """Switch between tabs in the application"""
    st.session_state.active_tab = tab_name
    
def switch_page(page_name):
    """Switch between main pages in the application"""
    st.session_state.current_page = page_name

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