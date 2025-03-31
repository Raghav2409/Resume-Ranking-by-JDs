import re

class JobDescriptionAnalyzer:
    """Analyzes job descriptions for skill coverage and other metrics"""
    def __init__(self):
        self.categories = {
            'Technical Skills': ['python', 'java', 'javascript', 'c#', 'ruby', 'sql', 'aws', 'azure', 'cloud', 
                               'docker', 'kubernetes', 'api', 'database', 'git', 'linux', 'agile', 'devops', 
                               'ml', 'ai', 'analytics', 'full-stack', 'frontend', 'backend'],
            'Soft Skills': ['communication', 'leadership', 'teamwork', 'collaboration', 'problem-solving', 
                           'analytical', 'initiative', 'organizational', 'time management', 'interpersonal'],
            'Experience Level': ['year', 'years', 'senior', 'junior', 'mid-level', 'lead', 'manager', 'experience'],
            'Education': ['degree', 'bachelor', 'master', 'phd', 'certification', 'education'],
            'Tools & Technologies': ['jira', 'confluence', 'slack', 'github', 'gitlab', 'azure', 'jenkins', 
                                   'terraform', 'react', 'angular', 'vue', 'node'],
            'Domain Knowledge': ['finance', 'healthcare', 'retail', 'banking', 'insurance', 'technology', 
                               'manufacturing', 'telecom', 'e-commerce']
        }

    def analyze_text(self, text):
        """Analyze text for keyword coverage in each category"""
        if not text:
            return {category: 0.0 for category in self.categories}
            
        text = text.lower()
        scores = {}
        
        for category, keywords in self.categories.items():
            category_score = 0
            for keyword in keywords:
                count = len(re.findall(r'\b' + keyword + r'\b', text))
                category_score += count
            max_possible = len(keywords)
            scores[category] = min(category_score / max_possible, 1.0)
            
        return scores
    
    def compare_jd_versions(self, original_jd, enhanced_versions):
        """Compare the original JD with enhanced versions"""
        # Get scores for original
        original_scores = self.analyze_text(original_jd)
        
        # Get scores for each enhanced version
        enhanced_scores = {
            f'Version {i+1}': self.analyze_text(version)
            for i, version in enumerate(enhanced_versions)
        }
        
        # Combine all for comparison
        all_scores = {'Original': original_scores, **enhanced_scores}
        
        return all_scores
    
    def identify_skill_gaps(self, jd_text):
        """Identify potential skill gaps in a job description"""
        scores = self.analyze_text(jd_text)
        gaps = []
        
        # Look for categories with low coverage
        for category, score in scores.items():
            if score < 0.2:
                gaps.append({
                    'category': category,
                    'score': score,
                    'suggestion': f"Consider adding more {category.lower()} requirements"
                })
        
        return gaps
    
    def extract_key_requirements(self, jd_text):
        """Extract key requirements from job description text"""
        # Simple extraction based on common patterns
        requirements = []
        
        # Look for bullet points
        bullet_pattern = r'[•\-\*]\s*(.*?)(?=\n[•\-\*]|\n\n|$)'
        bullets = re.findall(bullet_pattern, jd_text, re.DOTALL)
        for bullet in bullets:
            requirement = bullet.strip()
            if len(requirement) > 10:  # Only include substantive items
                requirements.append(requirement)
        
        # Look for lines with "required" or "must have"
        required_pattern = r'(?:required|must have|essential)[^\n]*'
        required_lines = re.findall(required_pattern, jd_text.lower())
        for line in required_lines:
            requirements.append(line.strip())
        
        return list(set(requirements))  # Remove duplicates