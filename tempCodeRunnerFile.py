@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_input = request.form["symptoms"]
        user_input_vectorized = vectorizer.transform([user_input])
        predicted_disease = model.predict(user_input_vectorized)[0]
        precautions = disease_info.get(predicted_disease, {}).get("precautions", "No precautions available.")
        food = disease_info.get(predicted_disease, {}).get("food", "No food recommendations available.")
        exercise = disease_info.get(predicted_disease, {}).get("exercise", "No exercise recommendations available.")
        medicines = disease_info.get(predicted_disease, {}).get("medicines", "No medicines available.")
        
        return render_template("index.html", prediction=predicted_disease, precautions=precautions, food=food, exercise=exercise, medicines=medicines)

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True, port=8000)