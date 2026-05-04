import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('processed_data.csv')

# 1. Create Intersectional Groups
df['Identity'] = df['Gender'] + " | " + df['Caste']

# 2. Set up the figure
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Advanced AI Fairness Audit: Intersectional & Counterfactual Analysis', fontsize=20)

# Panel A: The Intersectional Heatmap
pivot = df.pivot_table(index='Gender', columns='Caste', values='NLP_Score', aggfunc='mean')
sns.heatmap(pivot, annot=True, cmap='RdYlGn', ax=axes[0,0])
axes[0,0].set_title('Intersectional NLP Match Scores')

# Panel B: Selection Rate Gaps
threshold = df['NLP_Score'].quantile(0.85)
df['Selected'] = (df['NLP_Score'] >= threshold).astype(int)
df.groupby('Identity')['Selected'].mean().sort_values().plot(kind='barh', ax=axes[0,1], color='skyblue')
axes[0,1].set_title('Selection Probability by Intersectional Identity')

# Panel C: Score Volatility (Counterfactual Risk)
sns.boxplot(x='Gender', y='NLP_Score', data=df, ax=axes[1,0], palette='Set2')
axes[1,0].set_title('Score Distribution Range (Counterfactual Variance)')

# Panel D: Fairness Frontier (Parity vs Performance)
sns.kdeplot(data=df, x='NLP_Score', hue='Caste', fill=True, ax=axes[1,1])
axes[1,1].set_title('Caste-Based Distribution Overlap')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('reports/figures/advanced_fairness_dashboard.png')
print("✅ Advanced Dashboard Generated!")