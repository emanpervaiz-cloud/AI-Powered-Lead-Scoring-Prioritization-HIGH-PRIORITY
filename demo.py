import json
from lead_scorer import LeadScorer

def run_demo():
    print("Initializing Lead Scorer...")
    scorer = LeadScorer("processed_leads.csv")
    
    # In a real scenario, you'd run this once:
    # scorer.train_rf() 

    # Mock leads to test
    leads = [
        {
            "name": "High Intent Lead",
            "data": {'Lead Value': 5000, 'Engagement Score': 8, 'notes_intent_score': 3, 'stage_encoded': 2, 'source_encoded': 1, 'status_encoded': 1},
            "notes": "Looking to start migration process immediately. Has budget ready."
        },
        {
            "name": "Low Intent Lead",
            "data": {'Lead Value': 500, 'Engagement Score': 1, 'notes_intent_score': 0, 'stage_encoded': 0, 'source_encoded': 5, 'status_encoded': 3},
            "notes": "Just browsing for information, maybe next year."
        }
    ]

    print("\n" + "="*50)
    print("AI LEAD SCORING RESULTS")
    print("="*50)

    for lead in leads:
        print(f"\nAnalyzing: {lead['name']}")
        result = scorer.get_combined_score(lead['data'], lead['notes'])
        print(f"Final Score: {result['final_score']}")
        print(f"Grade: {result['grade']}")
        print(f"Summary: {result['summary']}")

if __name__ == "__main__":
    run_demo()
