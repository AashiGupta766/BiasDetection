import pandas as pd

# Load the data
df = pd.read_csv('processed_data.csv')

# 1. Identify the 'Bias Gap' using Intersectional Groups
# Hum manually 'Avg_Score' naam ka column create kar rahe hain
intersectional_stats = df.groupby(['Gender', 'Caste'])['NLP_Score'].agg(Avg_Score='mean').reset_index()

# Ab column exist karta hai, toh idxmax() kaam karega
max_group = intersectional_stats.loc[intersectional_stats['Avg_Score'].idxmax()]
min_group = intersectional_stats.loc[intersectional_stats['Avg_Score'].idxmin()]

# 2. Calculate the 'Intersectional Fairness Ratio'
fairness_ratio = min_group['Avg_Score'] / max_group['Avg_Score']

print("--- FINAL PROJECT INSIGHTS ---")
print(f"Most Advantaged Group: {max_group['Gender']} | {max_group['Caste']} (Score: {max_group['Avg_Score']:.4f})")
print(f"Most Disadvantaged Group: {min_group['Gender']} | {min_group['Caste']} (Score: {min_group['Avg_Score']:.4f})")
print("-" * 30)
print(f"Intersectional Fairness Ratio: {fairness_ratio:.2f}")

# Fairness Check (Industry Standard: 0.80)
if fairness_ratio < 0.8:
    print("⚠️ STATUS: INTERSECTIONAL BIAS DETECTED")
    print("Recommendation: Further feature pruning or adversarial debiasing required.")
else:
    print("✅ STATUS: FAIR SYSTEM")
    print("The model meets the 80% legal threshold across all intersectional groups.")