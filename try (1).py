from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import whois
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
app = Flask(__name__)

# Load the dataset
df = pd.read_csv(r"C:\Users\negia\OneDrive\Desktop\mali\malicious_phish.csv")

# Data Preprocessing Function
def preprocess_data(df):
    df = df.dropna()  # Drop missing values
    df['type'] = df['type'].map({'benign': 0, 'defacement': 1, 'phishing': 2, 'malware': 3})  # Encode labels
    return df
def get_domain_age(domain):
    try:
        domain_info = whois.whois(domain)
        creation_date = domain_info.creation_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        age = (pd.Timestamp.now() - creation_date).days / 365
        return age
    except:
        return None
# Feature Engineering Function
def feature_engineering(df):
    df['url_length'] = df['url'].apply(len)
    df['num_digits'] = df['url'].apply(lambda x: sum(c.isdigit() for c in x))
    df['domain_age'] = df['url'].apply(lambda x: get_domain_age(x))
    return df
df = preprocess_data(df)
df = feature_engineering(df)

# Separate features and labels
X = df[['url_length', 'num_digits', 'domain_age']]  # Only include numeric features
y = df['type']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'malicious_url_model.pkl')

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Load the trained model
model = joblib.load('malicious_url_model.pkl')
model_path = 'model.pkl'
joblib.dump(model, model_path)
def predict_url(url):
    domain_age = get_domain_age(url)
    url_length = len(url)
    num_special_chars = sum([1 for c in url if not c.isalnum()])

    # Create a DataFrame for the input URL
    input_df = pd.DataFrame({
        'domain_age': [domain_age],
        'url_length': [url_length],
        'num_special_chars': [num_special_chars]
    })

    input_encoded = pd.get_dummies(input_df, drop_first=True)

    # Ensure the input has the same columns as the training data
    missing_cols = set(X_encoded.columns) - set(input_encoded.columns)
    for col in missing_cols:
        input_encoded[col] = 0

    input_encoded = input_encoded[X_encoded.columns]
    
    prediction = model.predict(input_encoded)
    return prediction[0]

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/wer')
def more():
    return render_template('wer.html')
@app.route('/login')
def login():
    return render_template('login.html')
@app.route('/index')
def index():
    return render_template('index.html', accuracy=accuracy, precision=precision, recall=recall, f1=f1)
@app.route('/feedback')
def feedback():
    return render_template('feedback.html')
@app.route('/api/analyze', methods=['POST'])
def analyze_url():
    data = request.json
    url = data.get('url')
    if not url:
        return jsonify({'status': 'error', 'message': 'URL is required'}), 400

    prediction = predict_url(url)
    return jsonify({'status': 'success', 'prediction': prediction})
@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted')
    }
    return jsonify({'status': 'success', 'metrics': metrics})
feedback_data = []

@app.route('/api/feedback', methods=['POST'])
def provide_feedback():
    data = request.json
    url = data.get('url')
    is_malicious = data.get('is_malicious')
    if not url or is_malicious is None:
        return jsonify({'status': 'error', 'message': 'URL and feedback are required'}), 400

    feedback_data.append({'url': url, 'is_malicious': is_malicious})
    return jsonify({'status': 'success', 'message': 'Feedback received'})

@app.route('/api/retrain', methods=['POST'])
def retrain_model():
    global model, X_encoded, y

    for feedback in feedback_data:
        url = feedback['url']
        is_malicious = feedback['is_malicious']
        
        domain_age = get_domain_age(url)
        url_length = len(url)
        num_special_chars = sum([1 for c in url if not c.isalnum()])

        new_data = pd.DataFrame({
            'domain_age': [domain_age],
            'url_length': [url_length],
            'num_special_chars': [num_special_chars]
        })

        new_data_encoded = pd.get_dummies(new_data, drop_first=True)

        # Ensure the new data has the same columns as the training data
        missing_cols = set(X_encoded.columns) - set(new_data_encoded.columns)
        for col in missing_cols:
            new_data_encoded[col] = 0

        new_data_encoded = new_data_encoded[X_encoded.columns]
        
        X_encoded = pd.concat([X_encoded, new_data_encoded], ignore_index=True)
        y = pd.concat([y, pd.Series(is_malicious)], ignore_index=True)
    
    # Retrain the model
    model.fit(X_encoded, y)
    joblib.dump(model, model_path)

    feedback_data.clear()
    return jsonify({'status': 'success', 'message': 'Model retrained with feedback data'})


@app.route('/predict', methods=['POST'])
def predict():
    url = request.form.get('url')
    if url:
        domain_age = get_domain_age(url)
        url_length = len(url)
        num_digits = sum(c.isdigit() for c in url)

        features = pd.DataFrame({
            'domain_age': [domain_age],
            'url_length': [url_length],
            'num_digits': [num_digits]
        })

        # Handle missing columns if any
        input_encoded = pd.get_dummies(features, drop_first=True)
        missing_cols = set(X.columns) - set(input_encoded.columns)
        for col in missing_cols:
            input_encoded[col] = 0

        input_encoded = input_encoded[X.columns]

        prediction = model.predict(input_encoded)[0]
        label_map = {0: 'benign', 1: 'defacement', 2: 'phishing', 3: 'malware'}
        result = label_map[prediction]
        return render_template('index.html', url=url, result=result, accuracy=accuracy, precision=precision, recall=recall, f1=f1)
    return render_template('index.html', accuracy=accuracy, precision=precision, recall=recall, f1=f1)


if __name__ == '__main__':
    app.run(debug=True)
