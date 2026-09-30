#dataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tkinter as tk

true_news = pd.read_csv("True.csv")
fake_news = pd.read_csv("Fake.csv")

print("True News:")
print(true_news.head())

print("\nFake News:")
print(fake_news.head())

#lables

true_news["label"] = "REAL"
fake_news["label"] = "FAKE"

data = pd.concat([true_news, fake_news], ignore_index=True)

print(data["label"].value_counts())

# Separate news text and label
X = data["text"].fillna("").astype(str)
y = data["label"]

print("News Text:")
print(X.head())

print("\nLabels:")
print(y.head())

from sklearn.model_selection import train_test_split

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:", len(X_train))
print("Testing data:", len(X_test))

# Convert text into numbers using TF-IDF
# Convert text into numbers using TF-IDF
vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

print("TF-IDF conversion completed")
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


model = LogisticRegression()

model.fit(X_train, y_train)

print("Model training completed")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("Accuracy in percentage:", accuracy * 100, "%")

window = tk.Tk()
window.title("Fake News Detector")
window.geometry("600x400")

heading = tk.Label(
    window,
    text="Fake News Detector",
    font=("Arial", 22)
)
heading.pack(pady=20)

instruction = tk.Label(
    window,
    text="Enter a news article below:",
    font=("Arial", 12)
)
instruction.pack()

news_box = tk.Text(
    window,
    height=8,
    width=60
)
news_box.pack(pady=10)

result = tk.Label(
    window,
    text="Result will appear here",
    font=("Arial", 14)
)
result.pack(pady=10)
def check_news():
    news = news_box.get("1.0", tk.END).strip()

    if news == "":
        result.config(text="Please enter a news article")
        return

    news_vector = vectorizer.transform([news])
    prediction = model.predict(news_vector)
    probability = model.predict_proba(news_vector)

    confidence = max(probability[0]) * 100

    if prediction[0] == "REAL":
        result.config(
            text=f"Prediction: REAL\nModel Confidence: {confidence:.2f}%",
            fg="green"
        )
    else:
        result.config(
            text=f"Prediction: FAKE\nModel Confidence: {confidence:.2f}%",
            fg="red"
        )
def clear_news():
    news_box.delete("1.0", tk.END)
    result.config(text="Result will appear here")


clear_button = tk.Button(
    window,
    text="CLEAR",
    command=clear_news,
    font=("Arial", 12)
)

clear_button.pack(pady=5)
predict_button = tk.Button(
    window,
    text="PREDICT",
    command=check_news,
    font=("Arial", 12)
)

predict_button.pack(pady=10)
###website
import joblib

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model and vectorizer saved successfully")


window.mainloop()














