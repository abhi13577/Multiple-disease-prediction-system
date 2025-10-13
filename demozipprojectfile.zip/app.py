# app.py

import joblib
import numpy as np
import pandas as pd
# --- Flask Core Imports ---
from flask import Flask, request, render_template, jsonify, redirect, url_for, session
# --- Utility Imports ---
import os
import sqlite3
import json 
import uuid 
# --- Scientific Imports (Matplotlib must be configured first) ---
import matplotlib
matplotlib.use('Agg') # Fixes RuntimeErrors when using Matplotlib in Flask threads
import matplotlib.pyplot as plt
import io
import base64
import shap
# --- Jinja/Markup Imports (Fixes ImportError and allows HTML in suggestions) ---
from markupsafe import Markup
from jinja2 import pass_context 
from google import genai
from google.genai import types


# --- Flask App Initialization ---
app = Flask(__name__, template_folder='templates')
# CRITICAL: Secret key for session management (used for page redirection)
app.secret_key = 'your_super_secret_key_for_sessions_12345' 

# --- Global Cache for Large Objects (SHAP Images) ---
IMAGE_CACHE = {} 
# -------------------------------------


# --- Database Setup ---
DB_NAME = 'user_reports.db'

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def log_prediction(disease, result, input_data):
    """Logs the prediction result and input parameters to the SQLite database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT id FROM users WHERE username = 'Current_User'")
    user_id_row = cursor.fetchone()
    user_id = user_id_row[0] if user_id_row else 1

    cursor.execute(
        "INSERT INTO reports (user_id, disease, prediction, input_data) VALUES (?, ?, ?, ?)",
        (user_id, disease, result, json.dumps(input_data)) 
    )
    conn.commit()
    conn.close()

@app.template_filter('from_json')
@pass_context 
def from_json_filter(context, value): 
    """Jinja filter to safely load JSON string data for the report."""
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}


# --- Model Loading and Setup ---
try:
    diabetes_model = joblib.load(os.path.join('models', 'diabetes_model.pkl'))
    diabetes_scaler = joblib.load(os.path.join('models', 'diabetes_scaler.pkl'))
    
    heart_model = joblib.load(os.path.join('models', 'heart_model.pkl'))
    heart_scaler = joblib.load(os.path.join('models', 'heart_scaler.pkl'))
    
    parkinsons_model = joblib.load(os.path.join('models', 'parkinsons_model.pkl'))
    parkinsons_scaler = joblib.load(os.path.join('models', 'parkinsons_scaler.pkl'))
    
    print("All models and scalers loaded successfully.")
except FileNotFoundError:
    print("Error: Model or scaler files not found. Please run 'train_models.py' first.")
    diabetes_model, diabetes_scaler, heart_model, heart_scaler, parkinsons_model, parkinsons_scaler = None, None, None, None, None, None

# Define the features for each model to ensure consistency
DIABETES_FEATURES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
HEART_FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
PARKINSONS_FEATURES = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']

# A small set of sample data is needed for SHAP's explainer masker.
diabetes_sample = np.zeros((1, len(DIABETES_FEATURES)))
heart_sample = np.zeros((1, len(HEART_FEATURES)))
parkinsons_sample = np.zeros((1, len(PARKINSONS_FEATURES)))

# --- SHAP EXPLANATION FUNCTION ---
def get_shap_explanation(model, features, feature_names, sample_data):
    """Calculates SHAP values for a single prediction and returns them as a dictionary."""
    
    explainer = shap.Explainer(model, sample_data)
    shap_values = explainer(features)
    
    if isinstance(shap_values.values, list):
        shap_values_list = shap_values.values[1][0].tolist() 
        base_value = shap_values.base_values[1]
    else:
        shap_values_list = shap_values.values[0].tolist()
        base_value = shap_values.base_values[0]

    if isinstance(base_value, np.ndarray):
        base_value = base_value.tolist()
        if isinstance(base_value, list):
            base_value = base_value[0]
    
    plt.figure()
    shap.summary_plot(shap_values, features, feature_names=feature_names, show=False)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    plt.close()
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    shap_data = {
        'base_value': float(base_value),
        'values': {
            name: value for name, value in zip(feature_names, shap_values_list)
        },
        'image': image_base64
    }
    return shap_data


# --- CHATBOT LOGIC ---
try:
    # DEBUG: Print status of API Key
    api_key_status = "Found" if os.getenv('GEMINI_API_KEY') else "NOT Found"
    print(f"Gemini API Key Status: {api_key_status}")

    client = genai.Client()
except Exception as e:
    client = None 

def get_gemini_response(message, prediction_result):
    if not client:
        return "Sorry, the AI service is currently unavailable. (API Key Error)"

    system_instruction = (
        "You are a helpful, professional, and empathetic medical assistant chatbot. "
        "ALWAYS preface your response with a disclaimer: 'Please remember, I am an AI and cannot provide medical advice. Consult a doctor for any health concerns.' "
        "Focus on answering the user's question, using the prediction result for context."
    )

    context = f"The user's most recent prediction result is: {prediction_result}."
    full_prompt = f"Context: {context}\nUser Question: {message}"

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[full_prompt],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction
            )
        )
        return response.text
    except Exception as e:
        print(f"Gemini API call failed: {e}")
        return "I apologize, but I encountered an error while processing your request."

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    current_prediction = data.get('prediction', 'No prediction yet.')
    
    bot_response = get_gemini_response(user_message, current_prediction)
    
    return jsonify({'response': bot_response})


# --- PAGE ROUTES ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/diabetes')
def diabetes_page():
    return render_template('diabetes.html')

@app.route('/heart')
def heart_page():
    return render_template('heart.html')

@app.route('/parkinsons')
def parkinsons_page():
    return render_template('parkinsons.html')


# --- PREDICTION AND REDIRECT ROUTES ---

@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    if diabetes_model is None:
        return render_template('diabetes.html', error="Prediction service temporarily unavailable.")
        
    data = request.form.to_dict()
    features = [float(data[key]) for key in DIABETES_FEATURES]
    scaled_features = diabetes_scaler.transform([features])
    prediction = diabetes_model.predict(scaled_features)[0]
    result = "Positive for Diabetes (Likely)" if prediction == 1 else "Negative for Diabetes (Unlikely)"
    
    explanation = get_shap_explanation(diabetes_model, scaled_features, DIABETES_FEATURES, diabetes_sample)
    log_prediction('Diabetes', result, data)
    
    # Generate unique ID and store image data in cache
    cache_id = str(uuid.uuid4())
    IMAGE_CACHE[cache_id] = explanation
    
    session['prediction_data'] = {
        'disease': 'Diabetes',
        'prediction': result,
        'image_id': cache_id, # Store ID instead of large image
        'inputs': data 
    }
    return redirect(url_for('render_results', disease='diabetes'))


@app.route('/predict_heart', methods=['POST'])
def predict_heart():
    if heart_model is None:
        return render_template('heart.html', error="Prediction service temporarily unavailable.")

    data = request.form.to_dict()
    features = [float(data[key]) for key in HEART_FEATURES]
    scaled_features = heart_scaler.transform([features])
    prediction = heart_model.predict(scaled_features)[0]
    result = "Positive for Heart Disease (Likely)" if prediction == 1 else "Negative for Heart Disease (Unlikely)"
    
    explanation = get_shap_explanation(heart_model, scaled_features, HEART_FEATURES, heart_sample)
    log_prediction('Heart Disease', result, data)
    
    cache_id = str(uuid.uuid4())
    IMAGE_CACHE[cache_id] = explanation

    session['prediction_data'] = {
        'disease': 'Heart Disease',
        'prediction': result,
        'image_id': cache_id,
        'inputs': data
    }
    return redirect(url_for('render_results', disease='heart'))


@app.route('/predict_parkinsons', methods=['POST'])
def predict_parkinsons():
    if parkinsons_model is None:
        return render_template('parkinsons.html', error="Prediction service temporarily unavailable.")
    
    data = request.form.to_dict()
    features = [float(data[key]) for key in PARKINSONS_FEATURES]
    scaled_features = parkinsons_scaler.transform([features])
    prediction = parkinsons_model.predict(scaled_features)[0]
    result = "Positive for Parkinson's (Likely)" if prediction == 1 else "Negative for Parkinson's (Unlikely)"

    explanation = get_shap_explanation(parkinsons_model, scaled_features, PARKINSONS_FEATURES, parkinsons_sample)
    log_prediction('Parkinsons', result, data)
    
    cache_id = str(uuid.uuid4())
    IMAGE_CACHE[cache_id] = explanation

    session['prediction_data'] = {
        'disease': 'Parkinsons',
        'prediction': result,
        'image_id': cache_id,
        'inputs': data
    }
    return redirect(url_for('render_results', disease='parkinsons'))


# --- NEW RESULTS ROUTE ---
@app.route('/results/<disease>')
def render_results(disease):
    data = session.pop('prediction_data', None)
    
    session_disease_name = data['disease'].lower().replace(" ", "").replace("'", "") if data and 'disease' in data else None
    
    if not data or session_disease_name != disease:
        return redirect(url_for('home')) 

    image_id = data.get('image_id')
    
    if image_id and image_id in IMAGE_CACHE:
        full_explanation = IMAGE_CACHE.pop(image_id) 
        data['explanation'] = full_explanation
    else:
        data['explanation'] = None 
        
    return render_template('results.html', data=data)


# --- ANALYSIS AND SUGGESTION ROUTE ---

@app.route('/report_analysis')
def report_analysis():
    conn = get_db_connection()
    reports = conn.execute('SELECT * FROM reports ORDER BY prediction_date DESC').fetchall()
    conn.close()
    
    if not reports:
        return render_template('report_history.html', analysis="No reports found.", suggestions=[Markup("Please submit a prediction first.")], history=[])

    # Convert SQLite Rows to a list of dictionaries
    reports_list = [dict(row) for row in reports]

    # 1. Convert to Pandas DataFrame for analysis
    df = pd.DataFrame(reports_list)
    
    # 2. Check and clean the 'input_data' column
    def safe_json_load(json_str):
        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            return {}
            
    df['parsed_input'] = df['input_data'].apply(safe_json_load)
    
    # --- ADVANCED ANALYSIS ---
    total_predictions = len(df)
    positive_results = df[df['prediction'].str.contains('Positive')].shape[0]
    most_common_disease = df['disease'].mode()[0] if not df['disease'].empty else 'N/A'
    
    # Get the latest report for suggestions
    latest_report = df.iloc[0]
    latest_inputs = latest_report['parsed_input']
    latest_disease = latest_report['disease']

    # 3. Risk Consistency Check
    risk_consistency = df[df['prediction'].str.contains('Positive') & 
                          (df['disease'] == latest_disease)].shape[0]

    # 4. Analyze Specific Inputs (Focusing on Glucose for Diabetes Example)
    glucose_status = ""
    if latest_disease == 'Diabetes' and 'Glucose' in latest_inputs:
        try:
            latest_glucose = float(latest_inputs['Glucose'])
            if latest_glucose > 125:
                glucose_status = f"Your latest Glucose ({latest_glucose}) is high. This is a primary risk factor."
            elif latest_glucose > 100:
                glucose_status = f"Your latest Glucose ({latest_glucose}) is slightly elevated. Monitor closely."
        except:
            pass 

    # --- ADVANCED SUGGESTIONS ---
    suggestions = []
    
    if latest_report['prediction'].find('Positive') != -1:
        suggestions.append(Markup(f"🚨 **High Priority:** Your latest result for **{latest_disease}** is Positive. Seek professional medical consultation immediately."))

    if risk_consistency >= 2:
        suggestions.append(Markup(f"🔴 **Persistent Risk:** You have received **{risk_consistency}** high-risk warnings for {latest_disease}. This suggests a real trend; prioritize lifestyle changes."))
    
    if glucose_status:
        suggestions.append(Markup(f"🩸 **Glucose Alert:** {glucose_status}"))
    
    if latest_report['prediction'].find('Negative') != -1 and total_predictions > 5:
        suggestions.append(Markup("✅ **Maintain Momentum:** All recent results are Negative. Continue your current healthy routine!"))
    
    if not suggestions:
        suggestions.append(Markup("👍 **All Clear:** Continue monitoring your health regularly."))

    # Prepare history for the frontend
    history_records = df[['prediction_date', 'disease', 'prediction', 'input_data']].head(10).to_dict('records')

    # Final Analysis Summary
    analysis = {
        'total': total_predictions,
        'positive': positive_results,
        'common': most_common_disease,
        'latest_disease': latest_disease,
    }

    return render_template('report_history.html', 
                           analysis=analysis,
                           suggestions=suggestions,
                           history=history_records)


# --- DEBUG ROUTE ---

@app.route('/debug/routes')
def debug_routes():
    if not app.debug:
        return jsonify({"error": "Debug routes are only available in debug mode."}), 403
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            'endpoint': rule.endpoint,
            'rule': str(rule),
            'methods': sorted([m for m in rule.methods if m not in ('HEAD','OPTIONS')])
        })
    routes = sorted(routes, key=lambda r: r['rule'])
    return jsonify(routes)


if __name__ == "__main__":
    # Ensure the database is initialized on startup
    import database
    database.init_db()
    app.run(debug=True)