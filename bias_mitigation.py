import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def run_mitigation():
    df = pd.read_csv('processed_data.csv')

    # NOVEL STEP: Neutralizing the JD
    # We remove "Expert", "Dominant", "Supportive" and keep only the core skills
    jd_neutral = "Python, Machine Learning, Data Science, SQL"
    
    # Text Processing
    df['Skills_Clean'] = df['Skills'].astype(str).str.replace("[", "").str.replace("]", "").str.replace("'", "")
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['Skills_Clean'])
    vec_neutral = vectorizer.transform([jd_neutral])

    # Calculate the new "Fair Score"
    df['Fair_Score'] = cosine_similarity(tfidf_matrix, vec_neutral).flatten()

    # Save to see the difference
    df.to_csv('processed_data.csv', index=False)
    
    # Quick Check: Compare the gap
    avg_masc = df['Score_Masculine_JD'].mean()
    avg_fair = df['Fair_Score'].mean()
    
    print("--- Mitigation Results ---")
    print(f"Original Masculine Score Avg: {avg_masc:.4f}")
    print(f"New Neutralized Score Avg: {avg_fair:.4f}")
    print("✅ Mitigation logic applied. The AI is now focusing only on skills.")

if __name__ == '__main__':
    run_mitigation()