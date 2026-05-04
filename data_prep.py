import pandas as pd
import numpy as np
import os

def prepare_data(input_path, output_path):
    df = pd.read_csv(input_path)
    
    # Logic to fill the missing values randomly
    df['Gender'] = np.random.choice(['Male', 'Female', 'Non-binary'], size=len(df), p=[0.48, 0.48, 0.04])
    df['Caste'] = np.random.choice(['General', 'OBC', 'SC', 'ST'], size=len(df), p=[0.40, 0.35, 0.18, 0.07])
    df['Gap_year'] = np.random.choice([0, 1, 2, 3], size=len(df), p=[0.70, 0.15, 0.10, 0.05])
    
    # Save to the new 'data' folder
    df.to_csv(output_path, index=False)
    print(f"✅ Step 1 Success: Data saved to {output_path}")

if __name__ == '__main__':
    prepare_data('cleaned_resume_data.csv', 'data/processed_data.csv')