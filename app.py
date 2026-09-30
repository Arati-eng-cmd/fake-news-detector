from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained model and TF-IDF vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    news = ""

    if request.method == "POST":

        news = request.form["news"]

        # Convert news into TF-IDF
        news_vector = vectorizer.transform([news])

        # Predict
        result = model.predict(news_vector)

        prediction = result[0]

        # Calculate confidence
        probability = model.predict_proba(news_vector)
        confidence = max(probability[0]) * 100

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        news=news

    )


if __name__ == "__main__":
    app.run(debug=True)