from flask import Flask, render_template, request
import joblib  # Changed from pickle to joblib for compressed files

app = Flask(__name__)

# Load all models - MODIFIED TO USE COMPRESSED FILES
models = {}
model_names = ['logistic_regression', 'random_forest', 'svm', 'xgboost']

for name in model_names:
    vectorizer, model = joblib.load(f'model/{name}_model_compressed.pkl')
    models[name] = model

# Rest of your existing code remains exactly the same...
common_vectorizer = vectorizer

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # ... (keep all your existing prediction logic)
    news_text = request.form['news']
    news_vector = common_vectorizer.transform([news_text])
    
    # Get predictions from all models
    predictions = {}
    votes = {'Real': 0, 'Fake': 0}
    
    for name, model in models.items():
        pred = model.predict(news_vector)[0]
        result = "Fake" if pred == 1 else "Real"
        predictions[name.replace("_", " ").title()] = result
        votes[result] += 1
    
    # Determine final prediction (majority vote)
    final_prediction = max(votes, key=votes.get)
    
    # Calculate confidence percentage (models agreeing / total models)
    confidence = (votes[final_prediction] / len(models)) * 100
    
    # For unanimous decision (if you prefer this instead of majority vote):
    # if votes['Real'] == len(models):
    #     final_prediction = 'Real'
    #     confidence = 100
    # elif votes['Fake'] == len(models):
    #     final_prediction = 'Fake'
    #     confidence = 100
    # else:
    #     final_prediction = 'Uncertain'
    #     confidence = 0
    
    return render_template('index.html', 
                         prediction=final_prediction,
                         confidence=round(confidence, 1),
                         news_text=news_text,
                         individual_predictions=predictions)  # optional: if you want to keep individual predictions

if __name__ == '__main__':
    app.run(debug=True)