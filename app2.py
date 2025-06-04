from flask import Flask, request, render_template, send_file
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load pre-trained model and vectorizer
model = joblib.load("model/sentiment_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

@app.route('/')
def index():
    return render_template("bulk.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'csvfile' not in request.files:
        return "No file uploaded", 400

    file = request.files['csvfile']

    if file.filename == '':
        return "No file selected", 400

    try:
        # Read uploaded CSV file
        data = pd.read_csv(file)

        # Ensure there is a 'Review' column
        if 'Review' not in data.columns:
            return "The uploaded file must have a 'Review' column", 400

        # Transform and predict
        X = vectorizer.transform(data['Review'])
        data['Predicted Sentiment'] = model.predict(X)

        # Save result to a file
        output_path = "predicted_test_data.csv"
        data.to_csv(output_path, index=False)

        return send_file(output_path, as_attachment=True)

    except Exception as e:
        return f"An error occurred: {e}", 500

if __name__ == "__main__":
    app.run(debug=True, port=5002)
