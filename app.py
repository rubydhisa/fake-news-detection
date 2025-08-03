from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load all models
models = {}
model_names = ['logistic_regression', 'random_forest', 'svm', 'xgboost']

for name in model_names:
    with open(f'model/{name}_model.pkl', 'rb') as f:
        vectorizer, model = pickle.load(f)
        models[name] = model

# Use common vectorizer from one of the models (they're trained the same)
common_vectorizer = vectorizer

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
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