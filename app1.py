from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from langdetect import detect
from googletrans import Translator
from textblob import TextBlob

# Initialize Flask app
app = Flask(__name__)

# Load the model and tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Define sentiment labels
labels = ["negative", "neutral", "positive"]

# Initialize the Google Translator
translator = Translator()

def analyze_sentiment(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Apply softmax to get probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Convert probabilities to a dictionary of labels and their corresponding confidence percentages
    sentiment_confidence = {labels[i]: probs[0, i].item() * 100 for i in range(len(labels))}
    
    # Sort the emotions by their confidence in descending order
    sorted_sentiments = dict(sorted(sentiment_confidence.items(), key=lambda item: item[1], reverse=True))
    
    # Identify the main emotion (the one with the highest confidence)
    main_emotion = max(sorted_sentiments, key=sorted_sentiments.get)
    main_emotion_confidence = sorted_sentiments[main_emotion]
    
    return main_emotion, main_emotion_confidence, sorted_sentiments

@app.route("/", methods=["GET", "POST"])
def index():
    main_emotion = None
    main_emotion_confidence = None
    sentiment_confidence = None
    translated_text = None
    original_text = None

    if request.method == "POST":
        # Get the input sentence from the form
        text = request.form["text"]
        original_text = text

        # Detect the language of the input text
        detected_language = detect(text)

        # Translate to English if necessary
        if detected_language != "en":
            translated_text = translator.translate(text, src=detected_language, dest="en").text
        else:
            translated_text = text

        # Check if the translated text is a single word
        if len(translated_text.split()) == 1:
            # Use TextBlob for sentiment analysis of single words
            blob = TextBlob(translated_text)
            polarity = blob.sentiment.polarity
            main_emotion = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
            main_emotion_confidence = abs(polarity) * 100  # Confidence as the magnitude of polarity
            sentiment_confidence = {
                "positive": main_emotion_confidence if main_emotion == "Positive" else 0,
                "neutral": 100 - main_emotion_confidence if main_emotion == "Neutral" else 0,
                "negative": 100 - main_emotion_confidence if main_emotion == "Negative" else 0
            }
        else:
            # Perform sentiment analysis on the (possibly translated) text using the pre-trained model
            main_emotion, main_emotion_confidence, sentiment_confidence = analyze_sentiment(translated_text)

    return render_template("app1_index.html", 
                           main_emotion=main_emotion,
                           main_emotion_confidence=main_emotion_confidence, 
                           sentiment_confidence=sentiment_confidence,
                           original_text=original_text,
                           translated_text=translated_text)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
