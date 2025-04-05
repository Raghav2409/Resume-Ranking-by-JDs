import streamlit as st

def set_page_config():
    """Configure the Streamlit page settings"""
    st.set_page_config(
        page_title="JD Agent",
        page_icon="💼",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

# Custom CSS for styling - darker theme with higher contrast
custom_css = """
<style>
    /* Base Text and Background Colors */
    .main .block-container {
        color: #E2E8F0;
        background-color: #1A202C;
    }
    
    /* Headers */
    .main-header {
        font-size: 26px;
        font-weight: bold;
        margin-bottom: 15px;
        color: #90CDF4;
    }
    
    .section-header {
        font-size: 22px;
        font-weight: bold;
        margin-top: 25px;
        margin-bottom: 10px;
        color: #90CDF4;
        padding: 5px 0;
        border-bottom: 2px solid #4299E1;
    }
    
    .subsection-header {
        font-size: 18px;
        font-weight: bold;
        margin-top: 15px;
        margin-bottom: 5px;
        color: #63B3ED;
    }
    
    /* Message Boxes */
    .highlight-box {
        background-color: #2D3748;
        border-left: 4px solid #4299E1;
        padding: 12px;
        margin: 12px 0;
        border-radius: 4px;
        color: #E2E8F0;
    }
    
    .success-box {
        background-color: #1C4532;
        border-left: 4px solid #38A169;
        padding: 12px;
        margin: 12px 0;
        border-radius: 4px;
        color: #C6F6D5;
    }
    
    .warning-box {
        background-color: #744210;
        border-left: 4px solid #ED8936;
        padding: 12px;
        margin: 12px 0;
        border-radius: 4px;
        color: #FEFCBF;
    }
    
    /* Content Cards */
    .tab-header {
        font-weight: bold;
        color: #63B3ED;
    }
    
    .metric-card {
        background-color: #2D3748;
        border-radius: 5px;
        padding: 12px;
        margin: 8px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        color: #E2E8F0;
        border: 1px solid #4A5568;
    }
    
    /* Category Boxes */
    .category-high {
        background-color: #1C4532;
        border-left: 3px solid #38A169;
        padding: 12px;
        margin: 8px 0;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
        color: #C6F6D5;
    }
    
    .category-medium {
        background-color: #744210;
        border-left: 3px solid #ED8936;
        padding: 12px;
        margin: 8px 0;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
        color: #FEFCBF;
    }
    
    .category-low {
        background-color: #742A2A;
        border-left: 3px solid #F56565;
        padding: 12px;
        margin: 8px 0;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
        color: #FED7D7;
    }
    
    /* Input Fields and Selectors */
    .stTextInput > div > div > input {
        color: #E2E8F0;
        border: 1px solid #4A5568;
        background-color: #2D3748;
    }
    
    .stSelectbox > div > div > div {
        color: #E2E8F0;
        border: 1px solid #4A5568;
        background-color: #2D3748;
    }
    
    .stTextArea > div > div > textarea {
        color: #E2E8F0;
        border: 1px solid #4A5568;
        background-color: #2D3748;
    }
    
    /* Buttons */
    .stButton > button {
        font-weight: 600;
        border-radius: 4px;
        background-color: #3182CE;
        color: white;
    }
    
    .stButton > button[kind="secondary"] {
        background-color: #2D3748;
        color: #90CDF4;
        border: 1px solid #4299E1;
    }
    
    /* Drag and drop styles */
    .drag-container {
        padding: 15px;
        border: 2px dashed #4A5568;
        border-radius: 5px;
        margin-bottom: 12px;
        background-color: #2D3748;
    }
    
    .drag-item {
        padding: 12px;
        background-color: #1A202C;
        border: 1px solid #4A5568;
        border-radius: 5px;
        margin-bottom: 8px;
        cursor: grab;
        color: #E2E8F0;
    }
    
    .drag-item:hover {
        background-color: #2D3748;
        border-color: #63B3ED;
    }
    
    /* Feedback Box */
    .client-feedback-box {
        background-color: #2D3748;
        border-left: 4px solid #4299E1;
        padding: 15px;
        margin: 12px 0;
        border-radius: 4px;
        color: #E2E8F0;
    }
    
    /* Tab Styling */
    button[data-baseweb="tab"] {
        background-color: #2D3748;
        color: #90CDF4;
        border-bottom: 2px solid #4299E1;
        font-weight: 500;
    }
    
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #4A5568;
        color: #90CDF4;
        border-bottom: 2px solid #63B3ED;
        font-weight: 600;
    }
    
    /* Better contrast for DataFrames and tables */
    .stDataFrame {
        color: #E2E8F0;
    }
    
    .stDataFrame [data-testid="stTable"] {
        background-color: #2D3748;
        border: 1px solid #4A5568;
    }
    
    .stDataFrame th {
        background-color: #4A5568;
        color: #90CDF4;
        font-weight: 600;
    }
    
    .stDataFrame td {
        color: #E2E8F0;
        border-bottom: 1px solid #4A5568;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #90CDF4;
        background-color: #2D3748;
        border-radius: 4px;
    }
    
    .streamlit-expanderContent {
        background-color: #1A202C;
        border: 1px solid #4A5568;
        border-top: none;
        border-radius: 0 0 4px 4px;
        padding: 10px;
        color: #E2E8F0;
    }
    
    /* Text Area Improvements */
    .stTextArea > div > div > textarea {
        background-color: #2D3748;
        color: #E2E8F0;
        border: 1px solid #4A5568;
        font-family: monospace;
    }
    
    /* Add more contrast to radio buttons and checkboxes */
    .stRadio > div {
        color: #E2E8F0;
    }
    
    .stCheckbox > div > label {
        color: #E2E8F0;
    }
    
    /* Make tabs more visible */
    div[data-testid="stVerticalBlock"] div[data-testid="stHorizontalBlock"] button {
        background-color: #2D3748;
        color: #90CDF4;
    }
    
    div[data-testid="stVerticalBlock"] div[data-testid="stHorizontalBlock"] button:hover {
        background-color: #4A5568;
    }
</style>
"""

# AWS API credentials - using direct values or Streamlit secrets as fallback
AWS_CREDENTIALS = {
    "access_key": st.secrets["aws"]["access_key"] if "aws" in st.secrets else "",
    "secret_key": st.secrets["aws"]["secret_key"] if "aws" in st.secrets else "",
    "region": st.secrets["aws"]["region"] if "aws" in st.secrets else "us-east-1"
}