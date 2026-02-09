# Multiple Disease Prediction System

A Flask-based health prediction and analysis platform that provides ML-powered risk assessments for **Diabetes**, **Heart Disease**, and **Parkinson’s**. The application includes:

- Disease-specific prediction forms with ML inference
- SHAP-based model explanations with visualizations
- A Gemini-powered chatbot for contextual, user-friendly guidance
- Report history and automated insights based on past predictions

## Features

- **Multi-disease predictions** using pre-trained models and scalers stored under `models/`.
- **Explainable AI** with SHAP summary plots rendered per prediction.
- **Report analysis** that aggregates prediction history, flags trends, and generates suggestions.
- **Gemini chatbot** integration for empathetic, context-aware explanations.
- **SQLite persistence** for storing prediction inputs and results.

## Project Structure

```
.
├── README.md
└── demozipprojectfile.zip/
    ├── app.py
    ├── data/
    │   ├── diabetes.csv
    │   ├── heart_disease.csv
    │   └── parkinsons.csv
    ├── data_preprocessing.py
    ├── database.py
    ├── models/
    │   ├── diabetes_model.pkl
    │   ├── diabetes_scaler.pkl
    │   ├── heart_model.pkl
    │   ├── heart_scaler.pkl
    │   ├── parkinsons_model.pkl
    │   └── parkinsons_scaler.pkl
    ├── static/
    │   ├── Script.js
    │   └── style.css
    ├── templates/
    │   ├── index.html
    │   ├── diabetes.html
    │   ├── heart.html
    │   ├── parkinsons.html
    │   ├── report_analysis.html
    │   ├── report_history.html
    │   └── results.html
    ├── train_models.py
    └── user_reports.db
```

## Requirements

- Python 3.10+ recommended
- Core libraries (not exhaustive):
  - `flask`
  - `numpy`, `pandas`
  - `scikit-learn`
  - `joblib`
  - `matplotlib`
  - `shap`
  - `google-genai`

> **Note:** SHAP visualizations are generated with Matplotlib using a non-interactive backend (`Agg`) for compatibility with Flask.

## Setup

1. **Install dependencies** (example using pip):

   ```bash
   pip install flask numpy pandas scikit-learn joblib matplotlib shap google-genai
   ```

2. **Set the Gemini API key** (optional, for chatbot functionality):

   ```bash
   export GEMINI_API_KEY="your_api_key"
   ```

3. **Run the application**:

   ```bash
   cd demozipprojectfile.zip
   python app.py
   ```

   The app will start in debug mode and initialize the SQLite database automatically.

## Usage

- Visit `/` for the landing page.
- Use the disease-specific forms (`/diabetes`, `/heart`, `/parkinsons`) to submit inputs.
- After a prediction, you will be redirected to `/results/<disease>`.
- Visit `/report_analysis` for aggregated reports and insights.
- Use the embedded chatbot (front-end) to ask questions about your latest result.

## Training Models (Optional)

If you want to retrain models using the provided datasets:

```bash
cd demozipprojectfile.zip
python train_models.py
```

This will:

- preprocess each dataset
- fit multiple models
- select the best model
- save the trained model + scaler to `models/`

## Security Notes

- The Flask session secret key is hardcoded for development. Update it before deploying.
- The Gemini chatbot requires a valid API key and will return a fallback message if missing.

## License

No license file is provided. Add one if you plan to distribute or commercialize this project.
