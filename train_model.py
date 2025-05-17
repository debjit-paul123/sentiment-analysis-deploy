import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

# Load training data
training_data = pd.read_csv("train_text (1).csv")
X = training_data['Review']
y = training_data['Sentiment']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=10000, stop_words="english", ngram_range=(1, 2))
X_tfidf = vectorizer.fit_transform(X)

# Split data into train and test
X_train, X_val, y_train, y_val = train_test_split(X_tfidf, y, test_size=0.2, stratify=y, random_state=42)

# Define the Random Forest model
rf_model = RandomForestClassifier(random_state=42, class_weight="balanced")

# Hyperparameter tuning using GridSearchCV
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "max_features": ["auto", "sqrt", "log2"],
}
grid_search = GridSearchCV(
    rf_model,
    param_grid,
    cv=StratifiedKFold(n_splits=5),
    scoring="f1_weighted",
    verbose=1,
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)

# Best model from Grid Search
best_rf_model = grid_search.best_estimator_

# Evaluate on validation set
y_pred = best_rf_model.predict(X_val)
print("Validation Performance:")
print(classification_report(y_val, y_pred))

# Save the model and vectorizer
joblib.dump(best_rf_model, "model/sentiment_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")
