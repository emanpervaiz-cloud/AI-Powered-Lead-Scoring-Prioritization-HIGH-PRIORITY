import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder

def preprocess_leads(file_path):
    # Read the CSV (handle potential encoding issues)
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

    print(f"Loaded {len(df)} rows.")

    # 1. Feature Selection & Cleaning
    features = [
        'Opportunity Name', 'stage', 'Lead Value', 'source', 
        'Engagement Score', 'status', 'Interested Service', 
        'Current Visa', 'Highest Qualification', 'Notes'
    ]
    
    # Filter to available columns (CSV has slightly different names sometimes)
    actual_cols = [col for col in features if col in df.columns]
    df = df[actual_cols].copy()

    # 2. Define "Hot Lead" Label (Ground Truth for testing)
    # Strategy: Leads in later stages (COE Received, Appointment Booked, Post Consultation) are likely "Hot"
    hot_stages = ['COE Received', 'Appointment Booked', 'Post Consultation']
    df['is_hot'] = df['stage'].apply(lambda x: 1 if x in hot_stages else 0)

    # 3. Numeric Preprocessing
    df['Lead Value'] = pd.to_numeric(df['Lead Value'], errors='coerce').fillna(0)
    df['Engagement Score'] = pd.to_numeric(df['Engagement Score'], errors='coerce').fillna(0)

    # 4. Keyword Extraction from Notes (Qualitative Signal for RF)
    def extract_notes_features(text):
        if not isinstance(text, str): return 0
        text = text.lower()
        # Keywords indicating high intent
        hot_keywords = ['wants to book', 'interested in pr', 'migration', 'urgent', 'pay', 'visa expiring']
        score = sum(1 for word in hot_keywords if word in text)
        return score

    df['notes_intent_score'] = df['Notes'].apply(extract_notes_features)

    # 5. Categorical Encoding
    le = LabelEncoder()
    categorical_cols = ['stage', 'source', 'status', 'Interested Service', 'Current Visa']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col + '_encoded'] = le.fit_transform(df[col])

    # 6. Final Feature Matrix for RF
    encoded_features = [
        'Lead Value', 'Engagement Score', 'notes_intent_score',
        'stage_encoded', 'source_encoded', 'status_encoded'
    ]
    
    print("\nFeature Matrix Sample:")
    print(df[encoded_features + ['is_hot']].head())
    
    return df

if __name__ == "__main__":
    csv_path = "opportunities.csv"
    processed_df = preprocess_leads(csv_path)
    if processed_df is not None:
        processed_df.to_csv("processed_leads.csv", index=False)
        print("\nProcessed data saved to processed_leads.csv")
