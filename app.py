import sqlite3
import os
import pandas as pd
import joblib
from flask import Flask, request, render_template, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Define base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "users.db")

# Load dataset
df = pd.read_csv(os.path.join(BASE_DIR, "disease_symptoms_binary.csv"))

# Features and labels
X = df.iloc[:, 1:]
y = df.iloc[:, 0]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = {
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=200)
}

accuracies = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")

# Select best model
best_model_name = max(accuracies, key=accuracies.get)
best_model = models[best_model_name]
print(f"\n✅ Best Model: {best_model_name} with Accuracy: {accuracies[best_model_name]:.4f}")

# Save best model
joblib.dump((X.columns.tolist(), best_model, best_model_name), os.path.join(BASE_DIR, "best_disease_model.pkl"))

# Load model
symptoms_list, best_model, best_model_name = joblib.load(os.path.join(BASE_DIR, "best_disease_model.pkl"))

# Flask app
app = Flask(__name__)
app.secret_key = "your_secret_key"

# Create users.db if it doesn't exist
def init_user_db():
    if not os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        conn.execute('CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT UNIQUE, password TEXT)')
        conn.commit()
        conn.close()

init_user_db()

# Home Page
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

# Login Page
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        conn = sqlite3.connect(DB_PATH)
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        conn.close()

        if user and check_password_hash(user[2], password):
            session["user"] = email
            return redirect(url_for("home"))
        else:
            flash("Invalid email or password")
            return redirect(url_for("login"))

    return render_template("login.html")

# Register Page
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email")
        password = generate_password_hash(request.form.get("password"))

        try:
            conn = sqlite3.connect(DB_PATH)
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
