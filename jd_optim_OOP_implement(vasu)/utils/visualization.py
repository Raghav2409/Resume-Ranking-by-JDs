import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def create_radar_chart(resume, job_desc):
    """Create a radar chart for skill matching visualization"""
    categories = ['Technical Skills', 'Tools Proficiency', 'Certifications']
    
    # Safely convert values to strings and handle missing values
    resume_skills = set(str(resume.get('Skills', '')).lower().split(', '))
    resume_tools = set(str(resume.get('Tools', '')).lower().split(', '))
    resume_certs = set(str(resume.get('Certifications', '')).lower().split(', '))
    
    job_skills = set(str(job_desc.get('Skills', '')).lower().split(', '))
    job_tools = set(str(job_desc.get('Tools', '')).lower().split(', '))
    
    # Calculate scores with more robust handling
    skill_score = len(resume_skills.intersection(job_skills)) / max(len(job_skills), 1) if job_skills else 0
    tools_score = len(resume_tools.intersection(job_tools)) / max(len(job_tools), 1) if job_tools else 0
    cert_score = min(len(resume_certs) / 10, 1.0)  # Normalize certification count (capped at 1.0)
    
    scores = [skill_score, tools_score, cert_score]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        name='Match Score'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        height=250,
        margin=dict(l=20, r=20, t=30, b=20),
        title=None
    )
    return fig

def create_multi_radar_chart(scores_dict):
    """Create a radar chart comparing multiple job descriptions"""
    categories = list(next(iter(scores_dict.values())).keys())
    
    fig = go.Figure()
    
    for label, scores in scores_dict.items():
        fig.add_trace(go.Scatterpolar(
            r=[scores.get(cat, 0) for cat in categories],
            theta=categories,
            fill='toself',
            name=label
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title="Job Description Comparison",
        height=600
    )
    
    return fig

def create_distribution_chart(categorized_resumes):
    """Create a distribution chart showing resume categories"""
    categories = ['High Match', 'Medium Match', 'Low Match']
    counts = [
        len(categorized_resumes.get('high_matches', [])),
        len(categorized_resumes.get('medium_matches', [])),
        len(categorized_resumes.get('low_matches', []))
    ]
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=counts,
            marker_color=['#2ecc71', '#f39c12', '#e74c3c']
        )
    ])
    
    fig.update_layout(
        title="Match Distribution",
        xaxis_title=None,
        yaxis_title="Count",
        height=200,
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=False
    )
    
    return fig

def create_comparison_dataframe(scores_dict):
    """Create a DataFrame comparing multiple job descriptions"""
    if not scores_dict:
        return pd.DataFrame()
        
    categories = list(next(iter(scores_dict.values())).keys())
    
    df_data = {
        'Category': categories,
    }
    
    # Add scores for each version
    for label, scores in scores_dict.items():
        df_data[label] = [f"{scores.get(cat, 0):.2%}" for cat in categories]
        
        # Calculate change from original if this isn't the original
        if label != 'Original' and 'Original' in scores_dict:
            original_scores = scores_dict['Original']
            df_data[f'{label} Change'] = [
                f"{(scores.get(cat, 0) - original_scores.get(cat, 0))*100:+.2f}%" 
                for cat in categories
            ]
    
    return pd.DataFrame(df_data)