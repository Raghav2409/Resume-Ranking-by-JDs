import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Make sure NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def extract_skills(text):
    """Extract technical skills and technologies from text"""
    tech_keywords = {
        'programming_languages': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'scala', 'swift', 'golang'],
        'frameworks': ['django', 'flask', 'spring', 'react', 'angular', 'vue', 'nodejs', 'express', 'hibernate'],
        'databases': ['sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'redis', 'elasticsearch'],
        'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'jenkins'],
        'tools': ['git', 'maven', 'gradle', 'junit', 'selenium', 'jira', 'confluence']
    }
    
    text = str(text).lower()
    found_skills = {category: [] for category in tech_keywords}
    
    for category, keywords in tech_keywords.items():
        for keyword in keywords:
            if re.search(r'\b' + keyword + r'\b', text):
                found_skills[category].append(keyword)
    
    return found_skills

def preprocess_text(text):
    """Preprocess text for similarity comparison"""
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)

def detect_jd_type(file_name):
    """Detect the job description type based on the file name"""
    file_name = str(file_name).lower()
    
    # Define keyword patterns for each type
    java_python_keywords = ['java', 'python', 'support']
    data_engineer_keywords = ['data', 'analytics', 'aiml']
    
    # Check for Java/Python developer
    if any(keyword in file_name for keyword in java_python_keywords):
        return 'java_developer'
    
    # Check for Data Engineer
    elif any(keyword in file_name for keyword in data_engineer_keywords):
        return 'data_engineer'
    
    # Default type
    return 'general'