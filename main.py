from flask import Flask, render_template, request
import os

app = Flask(__name__, template_folder="templates")

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/english", methods=["GET", "POST"])
def english():
    if request.method == "POST":
        text = request.form["text"]
        # Do sentiment analysis...
        # Example result placeholders:
        sentiment_confidence = {
            'joy': 60.5,
            'sadness': 20.0,
            'anger': 15.5,
            'fear': 4.0
        }
        main_emotion = max(sentiment_confidence, key=sentiment_confidence.get)
        main_emotion_confidence = sentiment_confidence[main_emotion]
        return render_template("english.html", text=text, sentiment_confidence=sentiment_confidence, main_emotion=main_emotion, main_emotion_confidence=main_emotion_confidence)
    return render_template("english.html")


@app.route("/other_languages", methods=["GET", "POST"])
def other_languages():
    if request.method == "POST":
        text = request.form["text"]
        # translation and sentiment analysis logic
        original_text = text
        translated_text = "translated text"  # placeholder
        sentiment_confidence = {
            "joy": 50.0,
            "sadness": 25.0,
            "anger": 15.0,
            "fear": 10.0
        }
        main_emotion = max(sentiment_confidence, key=sentiment_confidence.get)
        main_emotion_confidence = sentiment_confidence[main_emotion]
        return render_template("other_languages.html", original_text=original_text, translated_text=translated_text, sentiment_confidence=sentiment_confidence, main_emotion=main_emotion, main_emotion_confidence=main_emotion_confidence)
    return render_template("other_languages.html")


@app.route('/emoji', methods=['GET', 'POST'])
def emoji():
    if request.method == 'POST':
        text = request.form.get('text')
        # Do emoji sentiment analysis...
        return render_template('emoji.html', result="Result")
    return render_template('emoji.html')

@app.route('/bulk', methods=['GET', 'POST'])
def bulk():
    if request.method == 'POST':
        file = request.files.get('file')
        # Process CSV file...
        return render_template('bulk.html', result="Uploaded successfully")
    return render_template('bulk.html')


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
