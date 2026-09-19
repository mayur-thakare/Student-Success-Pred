from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
import os
import io

from utils.file_processor import process_uploaded_file


app = Flask(__name__)

# Load trained model
MODEL_PATH = "student_performance_ridge_model.pkl"
model = joblib.load(MODEL_PATH)

# Required model columns
FEATURE_COLUMNS = [
    "Gender",
    "Study_Hours_per_Week",
    "Attendance_Rate",
    "Past_Exam_Scores",
    "Parental_Education_Level",
    "Internet_Access_at_Home",
    "Extracurricular_Activities"
]


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    error = None

    if request.method == "POST":

        try:
            gender = request.form["Gender"]
            study_hours = float(request.form["Study_Hours_per_Week"])
            attendance = float(request.form["Attendance_Rate"])
            past_score = float(request.form["Past_Exam_Scores"])
            parental_education = request.form["Parental_Education_Level"]
            internet_access = request.form["Internet_Access_at_Home"]
            extracurricular = request.form["Extracurricular_Activities"]

            student_data = pd.DataFrame({
                "Gender": [gender],
                "Study_Hours_per_Week": [study_hours],
                "Attendance_Rate": [attendance],
                "Past_Exam_Scores": [past_score],
                "Parental_Education_Level": [parental_education],
                "Internet_Access_at_Home": [internet_access],
                "Extracurricular_Activities": [extracurricular]
            })

            prediction = model.predict(student_data)[0]

        except Exception as e:
            error = f"Prediction error: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction,
        error=error
    )


@app.route("/upload", methods=["POST"])
def upload_file():

    try:

        uploaded_file = request.files.get("file")

        if not uploaded_file or uploaded_file.filename == "":
            return render_template(
                "index.html",
                error="Please select a file."
            )

        # Process uploaded file
        df = process_uploaded_file(uploaded_file)

        # Check required columns
        missing_columns = [
            column for column in FEATURE_COLUMNS
            if column not in df.columns
        ]

        if missing_columns:

            return render_template(
                "index.html",
                error=(
                    "Missing required columns: "
                    + ", ".join(missing_columns)
                )
            )

        # Keep only model features
        prediction_data = df[FEATURE_COLUMNS].copy()

        # Make predictions
        predictions = model.predict(prediction_data)

        # Add predictions to original data
        result_df = df.copy()

        result_df["Predicted_Final_Exam_Score"] = predictions.round(2)

        # Store result temporarily in memory
        output = io.BytesIO()

        result_df.to_excel(
            output,
            index=False,
            engine="openpyxl"
        )

        output.seek(0)

        return send_file(
            output,
            as_attachment=True,
            download_name="student_predictions.xlsx",
            mimetype=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            )
        )

    except Exception as e:

        return render_template(
            "index.html",
            error=f"File processing error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=True
    )