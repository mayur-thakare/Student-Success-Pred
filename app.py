from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from flask import Flask, render_template, request


# --------------------------------------------------
# Flask Application Setup
# --------------------------------------------------

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "student_performance_ridge_model.pkl"


# --------------------------------------------------
# Load Machine Learning Model
# --------------------------------------------------

try:
    model = joblib.load(MODEL_PATH)
    print("Student performance model loaded successfully.")

except Exception as error:
    model = None
    print("Error loading model:", error)


# --------------------------------------------------
# Prediction Category
# --------------------------------------------------

def get_performance_category(score):
    """
    Converts the predicted score into a performance category.
    """

    if score >= 80:
        return "Excellent"

    elif score >= 60:
        return "Good"

    elif score >= 40:
        return "Average"

    else:
        return "Needs Improvement"


# --------------------------------------------------
# Recommendations
# --------------------------------------------------

def generate_recommendations(
    study_hours,
    attendance,
    past_score,
    internet_access,
    extracurricular
):
    """
    Generates basic recommendations based on student inputs.
    """

    recommendations = []

    if study_hours < 10:
        recommendations.append(
            "Increase your weekly study hours and follow a consistent study timetable."
        )

    elif study_hours < 20:
        recommendations.append(
            "Try to gradually increase your study time for better academic performance."
        )

    else:
        recommendations.append(
            "Maintain your current study routine and revise regularly."
        )

    if attendance < 75:
        recommendations.append(
            "Improve your attendance because regular classes can help you understand concepts better."
        )

    else:
        recommendations.append(
            "Maintain your good attendance and participate actively in class."
        )

    if past_score < 50:
        recommendations.append(
            "Focus on weak subjects and practice previous examination questions."
        )

    elif past_score < 75:
        recommendations.append(
            "Practice mock tests and revise important topics to improve your scores."
        )

    else:
        recommendations.append(
            "Continue revising and solving advanced-level questions."
        )

    if internet_access == "No":
        recommendations.append(
            "Try to access educational resources through a library, college computer, or offline study material."
        )

    else:
        recommendations.append(
            "Use the internet for educational videos, online courses, and practice tests."
        )

    if extracurricular == "Yes":
        recommendations.append(
            "Balance extracurricular activities with your academic schedule."
        )

    else:
        recommendations.append(
            "Consider participating in suitable extracurricular activities for overall development."
        )

    return recommendations


# --------------------------------------------------
# Home Route
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    category = None
    recommendations = []
    error = None

    form_data = {
        "Gender": "",
        "Study_Hours_per_Week": "",
        "Attendance_Rate": "",
        "Past_Exam_Scores": "",
        "Parental_Education_Level": "",
        "Internet_Access_at_Home": "",
        "Extracurricular_Activities": ""
    }

    if request.method == "POST":

        try:
            # ------------------------------------------
            # Get Form Data
            # ------------------------------------------

            form_data["Gender"] = request.form.get("Gender", "")
            form_data["Study_Hours_per_Week"] = request.form.get(
                "Study_Hours_per_Week", ""
            )
            form_data["Attendance_Rate"] = request.form.get(
                "Attendance_Rate", ""
            )
            form_data["Past_Exam_Scores"] = request.form.get(
                "Past_Exam_Scores", ""
            )
            form_data["Parental_Education_Level"] = request.form.get(
                "Parental_Education_Level", ""
            )
            form_data["Internet_Access_at_Home"] = request.form.get(
                "Internet_Access_at_Home", ""
            )
            form_data["Extracurricular_Activities"] = request.form.get(
                "Extracurricular_Activities", ""
            )

            # ------------------------------------------
            # Validate Model
            # ------------------------------------------

            if model is None:
                raise Exception(
                    "Machine learning model could not be loaded."
                )

            # ------------------------------------------
            # Convert Numeric Values
            # ------------------------------------------

            study_hours = float(
                form_data["Study_Hours_per_Week"]
            )

            attendance = float(
                form_data["Attendance_Rate"]
            )

            past_score = float(
                form_data["Past_Exam_Scores"]
            )

            # ------------------------------------------
            # Basic Validation
            # ------------------------------------------

            if not 0 <= study_hours <= 100:
                raise ValueError(
                    "Study hours must be between 0 and 100."
                )

            if not 0 <= attendance <= 100:
                raise ValueError(
                    "Attendance rate must be between 0 and 100."
                )

            if not 0 <= past_score <= 100:
                raise ValueError(
                    "Past exam score must be between 0 and 100."
                )

            # ------------------------------------------
            # Create Input DataFrame
            # ------------------------------------------

            input_data = pd.DataFrame(
                [
                    {
                        "Gender": form_data["Gender"],
                        "Study_Hours_per_Week": study_hours,
                        "Attendance_Rate": attendance,
                        "Past_Exam_Scores": past_score,
                        "Parental_Education_Level": form_data[
                            "Parental_Education_Level"
                        ],
                        "Internet_Access_at_Home": form_data[
                            "Internet_Access_at_Home"
                        ],
                        "Extracurricular_Activities": form_data[
                            "Extracurricular_Activities"
                        ]
                    }
                ]
            )

            # ------------------------------------------
            # Make Prediction
            # ------------------------------------------

            predicted_value = model.predict(input_data)[0]

            prediction = round(float(predicted_value), 2)

            # Keep prediction between 0 and 100
            prediction = max(0, min(100, prediction))

            # ------------------------------------------
            # Category and Recommendations
            # ------------------------------------------

            category = get_performance_category(prediction)

            recommendations = generate_recommendations(
                study_hours=study_hours,
                attendance=attendance,
                past_score=past_score,
                internet_access=form_data[
                    "Internet_Access_at_Home"
                ],
                extracurricular=form_data[
                    "Extracurricular_Activities"
                ]
            )

        except ValueError as error_message:

            error = str(error_message)

        except Exception as error_message:

            error = (
                "Something went wrong while making the prediction: "
                + str(error_message)
            )

    return render_template(
        "index.html",
        prediction=prediction,
        category=category,
        recommendations=recommendations,
        form_data=form_data,
        error=error
    )


# --------------------------------------------------
# Run Application
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )