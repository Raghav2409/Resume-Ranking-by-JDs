# Resume Ranking System

A Python-based application for automatically ranking resumes against job descriptions using NLP techniques and AI-powered insights. The system provides two implementations:
1. A web interface built with Streamlit
2. A command-line version that outputs results to CSV

## Features

- Text preprocessing using NLTK
- TF-IDF vectorization for text comparison
- Cosine similarity scoring for resume-job matching
- AI-powered insights using OpenAI's GPT-3.5
- Top 3 resume matches for each job description
- Detailed match analysis including skills, tools, and certifications

## Prerequisites

- Python 3.7+
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/resume-ranking-system.git
cd resume-ranking-system
```

2. Set up your OpenAI API key:
   - Replace `"your-api-key-here"` in the code with your actual OpenAI API key
   - Alternatively, set it as an environment variable:
     ```bash
     export OPENAI_API_KEY='your-api-key-here'
     ```

## Input Data Format

The system expects two CSV files:

### Job Descriptions CSV
Required columns:
- `File Name`: Unique identifier for the job description
- `Skills`: Required skills for the position
- `Tools`: Required tools/technologies

### Resumes CSV
Required columns:
- `File Name`: Unique identifier for the resume
- `Skills`: Candidate's skills
- `Tools`: Tools/technologies the candidate is familiar with
- `Certifications`: Candidate's certifications

## Usage

### Streamlit Web Interface

1. Navigate to the project directory:
```bash
cd resume-ranking-system
```

2. Run the Streamlit app:
```bash
streamlit run frontend_JDOp.py
```

3. Access the web interface at `http://localhost:8501`

Features:
- Interactive job description selection
- Real-time resume ranking
- AI-generated insights for each match
- Visual presentation of results

### CSV Output Version

1. Update the `BASE_PATH` in `JDOp SaveToCSV.ipynb` to point to your data directory

The script will:
- Process all job descriptions against all resumes
- Generate match scores and AI insights
- Save results to a timestamped CSV file in the specified directory

Output format:
- Job Description
- Rank (1-3)
- Resume ID
- Match Score
- Skills Match
- Tools Match
- Certifications
- AI Insights

## How It Works

1. **Text Preprocessing**:
   - Converts text to lowercase
   - Removes stopwords
   - Applies lemmatization

2. **Similarity Scoring**:
   - Creates TF-IDF vectors for job descriptions and resumes
   - Computes cosine similarity between vectors
   - Ranks resumes based on similarity scores

3. **AI Insights**:
   - Analyzes matches using OpenAI's GPT-3.5
   - Provides detailed evaluation of candidate fit
   - Highlights candidate strengths

## File Structure

```
resume-ranking-system/
├── frontend_JDOp.py                           # Streamlit web interface
├── JDOp SaveToCSV.ipynb                       # CSV output version
├── README.md                                  # This file
├── resume_ranking_results_20250127_155107.csv #Top 3 resumes and their features for each JD
└── data dependencies/
    ├── job_descriptions_analysis_output.csv
    └── resumes_analysis_output.csv
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Acknowledgments

- NLTK for text processing
- scikit-learn for TF-IDF and similarity computation
- OpenAI for AI-powered insights
- Streamlit for the web interface
