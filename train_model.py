import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

# Load dataset
df = pd.read_csv('dataset/news.csv', quotechar='"')
df['label'] = df['label'].map({'real': 0, 'fake': 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Models to train
models = {
    "logistic_regression": LogisticRegression(),
    "random_forest": RandomForestClassifier(),
    "svm": SVC(probability=True),
    "xgboost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Create model directory
os.makedirs('model', exist_ok=True)

# Train and save each model
for name, clf in models.items():
    clf.fit(X_train_tfidf, y_train)
    y_pred = clf.predict(X_test_tfidf)
    print(f"{name} Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    with open(f'model/{name}_model.pkl', 'wb') as f:
        pickle.dump((vectorizer, clf), f)
