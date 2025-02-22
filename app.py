from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for

from flask_sqlalchemy import SQLAlchemy
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator
import emoji
import matplotlib.pyplot as plt
import io, base64, csv, datetime, json, requests, re, random
from wordcloud import WordCloud
from collections import Counter
import pandas as pd
import seaborn as sns

# Download required NLTK data
nltk.download('vader_lexicon')

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///sentiment_analysis.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

sia = SentimentIntensityAnalyzer()
app.secret_key = 'your_secret_key'  # Replace with a secure key in production

# Slack Webhook URL (replace with your actual webhook)
SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"

# List of profane words for toxicity detection
PROFANE_WORDS = ['fuck', 'shit', 'bitch', 'asshole', 'cunt', 'dick', 'piss']

# ---------------------------
# Database Model
# ---------------------------
class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    sentiment = db.Column(db.String(20), nullable=False)
    emojis = db.Column(db.String(200))
    scores = db.Column(db.Text)  # stored as JSON string
    timestamp = db.Column(db.String(50), nullable=False)

    def to_dict(self):
        return {
            "text": self.text,
            "sentiment": self.sentiment,
            "emojis": self.emojis,
            "scores": json.loads(self.scores),
            "timestamp": self.timestamp
        }

with app.app_context():
    db.create_all()

# ---------------------------
# Utility Functions
# ---------------------------
def translate_text(text):
    try:
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        return translated if translated else text
    except Exception as e:
        print(f"Translation Error: {e}")
        return text

def extract_emojis(text):
    return ''.join(c for c in text if c in emoji.EMOJI_DATA)

def analyze_emojis(emojis):
    positive_emojis = ['😊', '😃', '❤️', '👍', '😁', '🎉', '🥳', '💖', '🤩', '🤗', '😆']
    negative_emojis = ['😢', '😡', '💔', '👎', '😠', '😞', '😭', '😣', '😖', '😓', '😕']
    if any(e in emojis for e in positive_emojis):
        return "Positive"
    elif any(e in emojis for e in negative_emojis):
        return "Negative"
    return "Neutral"

def calculate_toxicity(text):
    text_lower = text.lower()
    count = 0
    for word in PROFANE_WORDS:
        count += len(re.findall(r'\b' + re.escape(word) + r'\b', text_lower))
    return count

def generate_sentiment_chart(scores):
    labels = ['Positive', 'Neutral', 'Negative']
    values = [scores['pos'], scores['neu'], scores['neg']]
    plt.figure(figsize=(5, 3))
    plt.bar(labels, values, color=['green', 'gray', 'red'])
    plt.title('Sentiment Breakdown')
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    chart_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return f"data:image/png;base64,{chart_url}"

def generate_wordcloud(text):
    try:
        wordcloud = WordCloud(width=400, height=200, background_color='white').generate(text)
        img = io.BytesIO()
        wordcloud.to_image().save(img, format='PNG')
        img.seek(0)
        wordcloud_url = base64.b64encode(img.getvalue()).decode()
        return f"data:image/png;base64,{wordcloud_url}"
    except Exception as e:
        print(f"WordCloud Generation Error: {e}")
        return None

def generate_emoji_chart(analyses):
    all_emojis = ''.join(entry.emojis for entry in analyses if entry.emojis)
    emoji_counts = Counter(all_emojis)
    if not emoji_counts:
        return None
    df = pd.DataFrame(emoji_counts.items(), columns=['Emoji', 'Count'])
    plt.figure(figsize=(6, 3))
    sns.barplot(x='Emoji', y='Count', data=df, palette='coolwarm', legend=False)
    plt.title('Most Used Emojis')
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    emoji_chart_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return f"data:image/png;base64,{emoji_chart_url}"

def send_slack_notification(text, sentiment):
    if sentiment == "Negative":
        message = {"text": f"Alert: Negative sentiment detected!\nText: {text}"}
        try:
            requests.post(SLACK_WEBHOOK_URL, json=message)
        except Exception as e:
            print(f"Slack Notification Error: {e}")

def predict_sentiment(text):
    translated_text = translate_text(text)
    emojis = extract_emojis(text)
    scores = sia.polarity_scores(translated_text)
    sentiment = ("Positive" if scores["compound"] >= 0.05
                 else "Negative" if scores["compound"] <= -0.05
                 else "Neutral")
    # Use emoji sentiment if available
    emoji_sentiment = analyze_emojis(emojis)
    if emoji_sentiment != "Neutral":
        sentiment = emoji_sentiment
    # Force negative if toxicity detected
    if calculate_toxicity(text) > 0:
        sentiment = "Negative"
    chart_url = generate_sentiment_chart(scores)
    wordcloud_url = generate_wordcloud(translated_text)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return sentiment, scores, emojis, chart_url, wordcloud_url, timestamp

# Dummy image sentiment prediction function (replace with your own model/API)
def predict_image_sentiment(image_file):
    sentiments = ["Positive", "Negative", "Neutral"]
    return random.choice(sentiments)

# ---------------------------
# Enterprise Feature Endpoints (Placeholders)
# ---------------------------
@app.route('/api-docs')
def api_docs():
    # Placeholder: Render API documentation page
    return render_template('api_docs.html')

@app.route('/explain', methods=['POST'])
def explain():
    # Placeholder: Return a dummy explanation for text analysis
    text = request.form.get('text', '')
    explanation = "Dummy Explanation: [Replace with LIME/SHAP output]"
    return jsonify({"explanation": explanation})

@app.route('/entities', methods=['POST'])
def entities():
    # Placeholder: Return dummy named entity recognition results
    text = request.form.get('text', '')
    entities = ["Entity1", "Entity2", "Entity3"]  # Replace with actual NER results
    return jsonify({"entities": entities})

@app.route('/login', methods=['GET', 'POST'])
def login():
    # Placeholder login endpoint
    if request.method == 'POST':
        # Dummy credential check
        session['user'] = request.form.get('username', 'demo_user')
        return redirect(url_for('home'))
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    # Placeholder registration endpoint
    if request.method == 'POST':
        # Process registration (dummy)
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/sandbox')
def sandbox():
    # Placeholder developer sandbox page
    return render_template('sandbox.html')
# ---------------------------
    
# ---------------------------
# Main Endpoints
# ---------------------------
@app.route('/')
def home():
    session.clear()  # Ensure fresh session
    analyses = Analysis.query.order_by(Analysis.id.desc()).all()
    emoji_chart_url = generate_emoji_chart(analyses)
    return render_template('home.html', history=[a.to_dict() for a in analyses], emoji_chart_url=emoji_chart_url)

@app.route('/predict', methods=['POST'])
def predict_route():
    text = request.form['text']
    sentiment, scores, emojis, chart_url, wordcloud_url, timestamp = predict_sentiment(text)
    analysis = Analysis(
        text=text,
        sentiment=sentiment,
        emojis=emojis,
        scores=json.dumps(scores),
        timestamp=timestamp
    )
    db.session.add(analysis)
    db.session.commit()
    send_slack_notification(text, sentiment)
    analyses = Analysis.query.order_by(Analysis.id.desc()).all()
    emoji_chart_url = generate_emoji_chart(analyses)
    return render_template('result.html', 
                           user_text=text, 
                           sentiment=sentiment, 
                           scores=scores, 
                           emojis=emojis, 
                           chart_url=chart_url, 
                           wordcloud_url=wordcloud_url, 
                           history=[a.to_dict() for a in analyses], 
                           emoji_chart_url=emoji_chart_url)

@app.route('/export')
def export():
    analyses = Analysis.query.order_by(Analysis.timestamp).all()
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(["Text", "Sentiment", "Emojis", "Scores", "Timestamp"])
    for a in analyses:
        cw.writerow([a.text, a.sentiment, a.emojis, a.scores, a.timestamp])
    output = io.BytesIO(si.getvalue().encode('utf-8'))
    output.seek(0)
    return send_file(output, mimetype="text/csv", as_attachment=True, download_name="sentiment_analysis.csv")

@app.route('/predict_image', methods=['GET', 'POST'])
def predict_image_route():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No image part", 400
        image_file = request.files['image']
        if image_file.filename == '':
            return "No selected file", 400
        sentiment = predict_image_sentiment(image_file)
        image_data = image_file.read()
        image_b64 = base64.b64encode(image_data).decode()
        return render_template('image_result.html', sentiment=sentiment, image_url="data:image/png;base64," + image_b64)
    return render_template('image.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.json
    text = data.get('text', '')
    sentiment, scores, emojis, chart_url, wordcloud_url, timestamp = predict_sentiment(text)
    return jsonify({
        'sentiment': sentiment, 
        'scores': scores, 
        'emojis': emojis, 
        'chart_url': chart_url, 
        'wordcloud_url': wordcloud_url, 
        'timestamp': timestamp
    })
# ---------------------------
# End of Main Endpoints
# ---------------------------

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
