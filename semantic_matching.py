import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def run_advanced_semantic_matching():
    # 1. Load the data
    df = pd.read_csv('C:\\Users\\asus\\Desktop\\bias-detection\\data\\processed_data.csv')
    
    # 2. Define two versions of the same Job Description
    # Version A: Masculine-coded (Words: 'Expert', 'Competitive', 'Lead', 'Dominant')
    jd_masculine = "Looking for an expert leader to dominate the market with competitive Python and ML skills. Must lead a high-performing team."
    
    # Version B: Feminine-coded (Words: 'Collaborative', 'Supportive', 'Growth', 'Community')
    jd_feminine = "Looking for a collaborative professional to support our growth. Help build a community using Python and ML skills in a supportive environment."

    # 3. Text Processing
    # Resumes are stored as lists of skills, let's make sure they are strings
    df['Skills_Clean'] = df['Skills'].astype(str).str.replace("[", "").str.replace("]", "").str.replace("'", "")
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['Skills_Clean'])

    # 4. Calculate Scores for both types of JDs
    vec_masculine = vectorizer.transform([jd_masculine])
    vec_feminine = vectorizer.transform([jd_feminine])

    df['Score_Masculine_JD'] = cosine_similarity(tfidf_matrix, vec_masculine).flatten()
    df['Score_Feminine_JD'] = cosine_similarity(tfidf_matrix, vec_feminine).flatten()

    # Create a final 'Standard_Score' for general audit (average of both)
    df['NLP_Score'] = (df['Score_Masculine_JD'] + df['Score_Feminine_JD']) / 2

    # 5. Save the updated database
    df.to_csv('processed_data.csv', index=False)
    print("✅ Semantic Matching Complete!")
    print(f"Sample Masculine Score: {df['Score_Masculine_JD'].mean():.4f}")
    print(f"Sample Feminine Score: {df['Score_Feminine_JD'].mean():.4f}")

if __name__ == '__main__':
    run_advanced_semantic_matching()
    