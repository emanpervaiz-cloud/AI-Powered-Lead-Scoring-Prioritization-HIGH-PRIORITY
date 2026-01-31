import pandas as pd
import joblib
import requests
import json
import google.generativeai as genai
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from config import GEMINI_API_KEY, OPENROUTER_API_KEY

class LeadScorer:
    def __init__(self, data_path="processed_leads.csv"):
        self.data_path = data_path
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.features = [
            'Lead Value', 'Engagement Score', 'notes_intent_score',
            'stage_encoded', 'source_encoded', 'status_encoded'
        ]
        
        # Initialize Gemini if available
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        else:
            self.gemini_model = None

    def train_rf(self):
        try:
            df = pd.read_csv(self.data_path)
        except Exception as e:
            print(f"Error loading training data: {e}")
            return

        X = df[self.features]
        y = df['is_hot']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"Training Random Forest on {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        print(f"RF Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        
        joblib.dump(self.model, "lead_scoring_rf_model.joblib")
        print("Model saved to lead_scoring_rf_model.joblib")

    def analyze_intent_ai(self, text):
        """
        Uses OpenRouter or Gemini to analyze qualitative intent.
        Returns a score (0-100) and extracted 'visa_eligibility'.
        """
        if not text or len(text) < 10:
            return {"intent_score": 0, "visa_probability": 0, "summary": "No data"}

        prompt = f"""
        Analyze the following lead notes for a migration consultancy service:
        "{text}"
        
        Provide a JSON response with:
        1. 'intent_score': (0-100) based on how likely they are to buy/enroll soon.
        2. 'visa_probability': (0-100) probability they are eligible for a visa based on details mentioned.
        3. 'summary': A 1-sentence summary of their needs.
        """

        # Try OpenRouter first if key is available
        if OPENROUTER_API_KEY:
            try:
                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    data=json.dumps({
                        "model": "google/gemini-2.0-flash-001", 
                        "messages": [{"role": "user", "content": prompt}],
                        "response_format": { "type": "json_object" }
                    })
                )
                result = response.json()
                return json.loads(result['choices'][0]['message']['content'])
            except Exception as e:
                print(f"OpenRouter Fallback: {e}")

        # Fallback to Gemini Pro directly
        if self.gemini_model:
            try:
                response = self.gemini_model.generate_content(prompt)
                return json.loads(response.text)
            except Exception as e:
                print(f"Gemini Error: {e}")

        return {"intent_score": 0, "visa_probability": 0, "summary": "AI processing failed"}

    def get_combined_score(self, lead_data, notes=None):
        """
        Merges RF prediction with AI qualitative analysis.
        """
        input_X = pd.DataFrame([lead_data])[self.features]
        # Ensure model is trained/loaded
        try:
            rf_prob = self.model.predict_proba(input_X)[0][1] * 100
        except:
            # Load from disk if not trained in session
            try:
                self.model = joblib.load("lead_scoring_rf_model.joblib")
                rf_prob = self.model.predict_proba(input_X)[0][1] * 100
            except:
                rf_prob = 50 # Default middle if no model
        
        # Get AI Insights
        ai_insights = self.analyze_intent_ai(notes)
        ai_score = ai_insights.get('intent_score', 0)
        
        # Weighted Merge (RF 60% / AI 40%)
        final_score = (rf_prob * 0.6) + (ai_score * 0.4)
        
        return {
            "final_score": int(final_score),
            "grade": "HOT" if final_score > 80 else "WARM" if final_score > 40 else "COLD",
            "visa_probability": ai_insights.get('visa_probability', 0),
            "summary": ai_insights.get('summary', "")
        }

if __name__ == "__main__":
    scorer = LeadScorer("processed_leads.csv")
    scorer.train_rf()
    
    # Test with a mock lead
    test_lead = {
        'Lead Value': 3500,
        'Engagement Score': 5,
        'notes_intent_score': 2,
        'stage_encoded': 1,
        'source_encoded': 5,
        'status_encoded': 2
    }
    test_notes = "Wants to book PR now, visa is expiring in 2 days. From Sydney."
    
    print("\n--- Testing Combined Score ---")
    result = scorer.get_combined_score(test_lead, test_notes)
    print(json.dumps(result, indent=2))
