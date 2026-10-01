from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        news = request.form.get('news', '')
        if news:
            vec = vectorizer.transform([news])
            pred = model.predict(vec)[0]
            prediction = "Fake News" if pred == 1 else "Real News"
    return render_template("index.html", prediction=prediction)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    return home()

if __name__ == "__main__":
    app.run()

        
