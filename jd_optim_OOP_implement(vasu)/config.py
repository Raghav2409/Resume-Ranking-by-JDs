import streamlit as st

def set_page_config():
    """Configure the Streamlit page settings"""
    st.set_page_config(
        page_title="JD Agent",
        page_icon="💼",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

# Custom CSS for styling
custom_css = """
<style>
    .main-header {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 15px;
        color: #1E3A8A;
    }
    
    .section-header {
        font-size: 20px;
        font-weight: bold;
        margin-top: 25px;
        margin-bottom: 10px;
        color: #1E3A8A;
    }
    
    .subsection-header {
        font-size: 16px;
        font-weight: bold;
        margin-top: 15px;
        margin-bottom: 5px;
        color: #2563EB;
    }
    
    .highlight-box {
        background-color: #F3F4F6;
        border-left: 4px solid #2563EB;
        padding: 10px;
        margin: 10px 0;
    }
    
    .success-box {
        background-color: #D1FAE5;
        border-left: 4px solid #10B981;
        padding: 10px;
        margin: 10px 0;
    }
    
    .warning-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 10px;
        margin: 10px 0;
    }
    
    .tab-header {
        font-weight: bold;
        color: #2563EB;
    }
    
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
        box-shadow: 1px 1px 5px rgba(0,0,0,0.1);
    }
    
    .category-high {
        background-color: #e6ffe6;
        border-left: 3px solid #2ecc71;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
    
    .category-medium {
        background-color: #fff5e6;
        border-left: 3px solid #f39c12;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
    
    .category-low {
        background-color: #ffe6e6;
        border-left: 3px solid #e74c3c;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
    
    /* Drag and drop styles */
    .drag-container {
        padding: 10px;
        border: 2px dashed #ddd;
        border-radius: 5px;
        margin-bottom: 10px;
        background-color: #f8f9fa;
    }
    
    .drag-item {
        padding: 10px;
        background-color: white;
        border: 1px solid #eee;
        border-radius: 5px;
        margin-bottom: 8px;
        cursor: grab;
    }
    
    .drag-item:hover {
        background-color: #f0f9ff;
        border-color: #bfdbfe;
    }
    
    .client-feedback-box {
        background-color: #f0f9ff;
        border-left: 4px solid #3b82f6;
        padding: 15px;
        margin: 10px 0;
        border-radius: 4px;
    }
    
    div[data-testid="stSidebarNav"] {
        background-color: #F3F4F6;
        padding-top: 2rem;
    }
</style>
"""

# AWS API credentials - properly formatted as a dictionary
AWS_CREDENTIALS = {
    "access_key": st.secrets["aws"]["access_key"] if "aws" in st.secrets else "",
    "secret_key": st.secrets["aws"]["secret_key"] if "aws" in st.secrets else "",
    "region": st.secrets["aws"]["region"] if "aws" in st.secrets else "us-east-1"