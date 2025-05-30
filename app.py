import sqlite3
import os
import pandas as pd
import joblib
from flask import Flask, request, render_template, jsonify, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load datasets
df = pd.read_csv("disease_symptoms_binary.csv")

# Extract features (symptoms) and labels (diseases)
X = df.iloc[:, 1:]
y = df.iloc[:, 0]

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=200)
}

# Train models and calculate accuracy
accuracies = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")

# Select the best model
best_model_name = max(accuracies, key=accuracies.get)
best_model = models[best_model_name]
print(f"\n✅ Best Model: {best_model_name} with Accuracy: {accuracies[best_model_name]:.4f}")

# Save the best model
joblib.dump((X.columns.tolist(), best_model, best_model_name), "best_disease_model.pkl")

# Load the trained best model
symptoms_list, best_model, best_model_name = joblib.load("best_disease_model.pkl")

# Flask Web App
app = Flask(__name__)
app.secret_key = "your_secret_key"  # required for login session

def init_user_db():
    if not os.path.exists("users.db"):
        conn = sqlite3.connect('users.db')
        conn.execute('CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT UNIQUE, password TEXT)')
        conn.commit()
        conn.close()

init_user_db()

# Disease Prediction Route
@app.route("/", methods=["GET", "POST"])
def home():
    if "user" not in session:
        return redirect(url_for("welcome"))

    if request.method == "POST":
        user_symptoms = request.form["symptoms"].split(", ")  
        input_vector = [1 if symptom in user_symptoms else 0 for symptom in symptoms_list]
        predicted_disease = best_model.predict([input_vector])[0]
        return render_template("index.html", prediction=predicted_disease, model_name=best_model_name)

    return render_template("index.html", prediction=None, model_name=None)


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = sqlite3.connect("users.db")
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        conn.close()

        if user and check_password_hash(user[2], password):
            session["user"] = email
            return redirect(url_for("home"))
        else:
            flash("Invalid email or password")
            return redirect(url_for("login"))

    return render_template("Login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form["email"]
        password = generate_password_hash(request.form["password"])

        try:
            conn = sqlite3.connect("users.db")
            conn.execute("INSERT INTO users (email, password) VALUES (?, ?)", (email, password))
            conn.commit()
            conn.close()
            flash("Registration successful! Please login.")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Email already exists.")
            return redirect(url_for("register"))

    return render_template("Register.html")


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

@app.route("/welcome")
def welcome():
    return render_template("welcome.html")

@app.route("/contact")
def contact():
    return render_template("ContactUs.html")

@app.route("/about")
def about():
    return render_template("Aboutus.html")

@app.route("/healthdetails")
def healthdetails():
    return render_template("HealthDetails.html")

if __name__ == "__main__":
    app.run(debug=True, port=8000)
