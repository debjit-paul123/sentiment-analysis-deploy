from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


app = Flask(__name__)


model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


labels = ["negative", "neutral", "positive"]


def analyze_sentiment_roberta(text):
    """ Analyze sentiment using Roberta model """
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    
    sentiment_confidence = {labels[i]: probs[0, i].item() * 100 for i in range(len(labels))}
    
    
    sorted_sentiments = dict(sorted(sentiment_confidence.items(), key=lambda item: item[1], reverse=True))
    
    
    main_emotion = max(sorted_sentiments, key=sorted_sentiments.get)
    main_emotion_confidence = sorted_sentiments[main_emotion]
    
    return main_emotion, main_emotion_confidence, sorted_sentiments


@app.route("/", methods=["GET", "POST"])
def index():
    main_emotion = None
    main_emotion_confidence = None
    sentiment_confidence = None
    text = None
    

    if request.method == "POST":
        
        text = request.form["text"]
        
            
        main_emotion, main_emotion_confidence, sentiment_confidence = analyze_sentiment_roberta(text)
    
    return render_template("app4_index.html", 
                           main_emotion=main_emotion,
                           main_emotion_confidence=main_emotion_confidence, 
                           sentiment_confidence=sentiment_confidence,
                           text=text)

if __name__ == "__main__":
    app.run(debug=True, port=5004)
