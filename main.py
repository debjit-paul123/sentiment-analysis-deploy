from flask import Flask, render_template, request
import os

app = Flask(__name__, template_folder="templates")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/english')
def english():
    return render_template('english.html')

@app.route('/other-languages')
def other_languages():
    return render_template('other_languages.html')

@app.route('/bulk')
def bulk():
    return render_template('bulk.html')

@app.route('/emoji')
def emoji():
    return render_template('emoji.html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
