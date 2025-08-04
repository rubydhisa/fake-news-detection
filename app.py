from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load all models and vectorizers
models = {}
model_names = ['logistic_regression', 'random_forest', 'svm', 'xgboost']

# Load common vectorizer (assuming it's the same for all models)
common_vectorizer = None

for name in model_names:
    try:
        # Load model and its corresponding vectorizer
        model_data = joblib.load(f'model/{name}_model_compressed.joblib')
        
        # First item is vectorizer, second is model
        vectorizer, model = model_data  
        models[name] = model
        
        # Set common vectorizer (assuming they're all the same)
        if common_vectorizer is None:
            common_vectorizer = vectorizer
            
    except Exception as e:
        print(f"Error loading {name} model: {str(e)}")
        continue

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'news' not in request.form:
        return render_template('index.html', error="No news text provided")
    
    news_text = request.form['news'].strip()
    if not news_text:
        return render_template('index.html', error="News text cannot be empty")
    
    try:
        # Vectorize the input text
        news_vector = common_vectorizer.transform([news_text])
        
        # Get predictions from all loaded models
        predictions = {}
        votes = {'Real': 0, 'Fake': 0}
        
        for name, model in models.items():
            try:
                pred = model.predict(news_vector)[0]
                result = "Fake" if pred == 1 else "Real"
                predictions[name.replace("_", " ").title()] = result
                votes[result] += 1
            except Exception as e:
                print(f"Error predicting with {name}: {str(e)}")
                continue
        
        # Determine final prediction (majority vote)
        final_prediction = max(votes, key=votes.get)
        confidence = (votes[final_prediction] / len(models)) * 100
        
        return render_template('index.html', 
                             prediction=final_prediction,
                             confidence=round(confidence, 1),
                             news_text=news_text,
                             individual_predictions=predictions)
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return render_template('index.html', error="Error processing your request")

if __name__ == '__main__':
    # Get port from environment variable or use 5000 as default
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)