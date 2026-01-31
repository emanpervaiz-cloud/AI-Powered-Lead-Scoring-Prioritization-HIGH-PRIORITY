# AI-Powered Lead Scoring & Prioritization (High Priority)

This repository contains the core logic for an AI-powered lead scoring system designed to prioritize high-value migration consultancy leads.

## Features
- **Machine Learning (Random Forest)**: Predicts lead "hotness" based on historical data (Lead Value, Engagement, etc.).
- **AI Qualitative Analysis**: Uses Gemini to analyze lead notes for intent and visa eligibility.
- **Hybrid Scoring**: Combines quantitative ML predictions with qualitative AI insights.

## Project Structure
- `lead_scorer.py`: Core AI logic.
- `data_prep.py`: Data cleaning and feature engineering.
- `demo.py`: Simple demonstration of the scoring system.
- `opportunities.csv`: Sample training data.
- `config.py`: Environment configuration.

## Setup
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your `GEMINI_API_KEY`.
4. Process data and train the model:
   ```bash
   python data_prep.py
   python lead_scorer.py
   ```
5. Run the demo:
   ```bash
   python demo.py
   ```
