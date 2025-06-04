from flask import Flask, request, render_template, jsonify
import emoji
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

# Load emoji sentiment data from CSV
emoji_data = pd.read_csv('Emoji_Sentiment_Data.csv')
emoji_sentiments = dict(zip(emoji_data['emoji'], emoji_data['sentiment']))

# Load sentiment analysis model
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

LABELS = ["negative", "neutral", "positive"]

def analyze_sentiment(text):
    # Extract emojis
    emojis_in_text = [char for char in text if emoji.is_emoji(char)]
    
    emoji_analysis = []
    if emojis_in_text:
        for em in emojis_in_text:
            sentiment = emoji_sentiments.get(em, "neutral")
            emoji_analysis.append({"emoji": em, "sentiment": sentiment})
    else:
        emoji_analysis = [{"error": "No emojis found in the text."}]
    
    # Text sentiment analysis
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    sentiment_index = torch.argmax(scores).item()
    text_sentiment = LABELS[sentiment_index]
    text_polarity = round(scores[sentiment_index].item() * 100, 2)

    return {
        "input_text": text,
        "text_sentiment": text_sentiment,
        "text_polarity": text_polarity,
        "emoji_analysis": emoji_analysis
    }

@app.route('/')
def home():
    return render_template('emoji.html')  # <-- this was the fix

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    if 'text' not in data or not data['text'].strip():
        return jsonify({"error": "No text provided or input is empty."}), 400
    
    text = data['text']
    sentiment = analyze_sentiment(text)
    return jsonify(sentiment)

if __name__ == '__main__':
    app.run(debug=True, port=5003)
