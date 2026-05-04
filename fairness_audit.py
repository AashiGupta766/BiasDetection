import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_fairness_audit():
    # 1. Load the updated data
    if not os.path.exists('processed_data.csv'):
        print("Error: processed_data.csv not found!")
        return
        
    df = pd.read_csv('processed_data.csv')

    # Create reports directory if it doesn't exist
    os.makedirs('reports/figures', exist_ok=True)

    # 2. Analyze Gender Bias across JD Types
    plt.figure(figsize=(14, 6))

    # Plot 1: Gender vs JD Type
    plt.subplot(1, 2, 1)
    gender_scores = df.groupby('Gender')[['Score_Masculine_JD', 'Score_Feminine_JD']].mean()
    gender_scores.plot(kind='bar', ax=plt.gca())
    plt.title('Avg Score by Gender & JD Language')
    plt.ylabel('Cosine Similarity Score')
    plt.xticks(rotation=0)

    # Plot 2: Caste vs JD Type
    plt.subplot(1, 2, 2)
    caste_scores = df.groupby('Caste')[['Score_Masculine_JD', 'Score_Feminine_JD']].mean()
    caste_scores.plot(kind='bar', ax=plt.gca())
    plt.title('Avg Score by Caste & JD Language')
    plt.ylabel('Cosine Similarity Score')
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.savefig('reports/figures/fairness_comparison.png')
    print("✅ Audit Complete! Charts saved to 'reports/figures/fairness_comparison.png'")

    # 3. Calculate Disparate Impact Ratio
    # We'll define 'Shortlisted' as top 20% of the NLP_Score
    threshold = df['NLP_Score'].quantile(0.80)
    df['Is_Shortlisted'] = (df['NLP_Score'] >= threshold).astype(int)

    print("\n--- Disparate Impact Analysis (Selection Rate) ---")
    for attr in ['Gender', 'Caste']:
        rates = df.groupby(attr)['Is_Shortlisted'].mean()
        ref_group = rates.idxmax()
        print(f"\nAttribute: {attr} (Reference Group: {ref_group})")
        for group, rate in rates.items():
            di_ratio = rate / rates[ref_group]
            status = "❌ BIASED" if di_ratio < 0.8 else "✅ FAIR"
            print(f" - {group}: {rate:.2%} (DI Ratio: {di_ratio:.2f}) -> {status}")

if __name__ == '__main__':
    run_fairness_audit()