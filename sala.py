import streamlit as st
import joblib
import numpy as np
from datetime import date
import os

st.set_page_config(page_title="Salary Prediction App", layout="centered")

st.title("💼 Salary Prediction App")
st.write("Enter employee details below to get an estimated salary.")

# make base directory robust when __file__ may be undefined
try:
    base_dir = os.path.dirname(__file__)
except NameError:
    base_dir = os.getcwd()
MODEL_PATH = os.path.join(base_dir, "linearmodel.pkl")


class FallbackModel:
    def predict(self, X):
        arr = np.asarray(X, dtype=float)
        # simple heuristic using columns: [age, sex_enc, experience, code_enc, qual_enc]
        age = arr[:, 0]
        exp = arr[:, 2]
        code = arr[:, 3]
        qual = arr[:, 4]
        return 20000 + age * 300 + exp * 1500 + code * 800 + qual * 1200

# --- Load model safely ---
@st.cache_resource
def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            st.info("Model file 'linearmodel.pkl' not found. Using a fallback predictor.")
            return {"model": FallbackModel(), "scaler": None, "label_encoders": None}
        data = joblib.load(MODEL_PATH)

        # Handle both cases: dict or model only
        if isinstance(data, dict):
            model = data.get("model", FallbackModel())
            scaler = data.get("scaler", None)
            label_encoders = data.get("label_encoders", None)
        else:
            model = data
            scaler = None
            label_encoders = None

        return {"model": model, "scaler": scaler, "label_encoders": label_encoders}
    except Exception as e:
        st.warning(f"Failed to load model, using fallback predictor: {e}")
        return {"model": FallbackModel(), "scaler": None, "label_encoders": None}

saved_data = load_model()
if saved_data is None:
    st.stop()

model = saved_data["model"]
scaler = saved_data["scaler"]
label_encoders = saved_data["label_encoders"]

# helper to safely get classes list
def classes_or_default(le_dict, key, default):
    if le_dict and key in le_dict:
        le = le_dict[key]
        if hasattr(le, "classes_"):
            return list(le.classes_)
    return default

# --- Input fields ---
name = st.text_input("Name")
age = st.number_input("Age", 18, 70, 25)
sex = st.selectbox("Sex", classes_or_default(label_encoders, "sex", ["Male", "Female"]))
dob = st.date_input("Date of Birth", date(2000, 1, 1))
experience = st.number_input("Experience (Years)", 0, 50, 2)
coding_language = st.selectbox("Coding Language", classes_or_default(label_encoders, "coding_language", ["Python", "Java", "C++", "JavaScript"]))
qualification = st.selectbox("Qualification", classes_or_default(label_encoders, "qualification", ["Bachelors", "Masters", "PhD"]))

# --- Predict section ---
if st.button("Predict Salary"):
    try:
        # Encode categorical fields safely
        if label_encoders and "sex" in label_encoders:
            sex_encoded = int(label_encoders["sex"].transform([sex])[0])
        else:
            sex_encoded = 1 if sex == "Male" else 0

        if label_encoders and "coding_language" in label_encoders:
            code_encoded = int(label_encoders["coding_language"].transform([coding_language])[0])
        else:
            code_encoded = 0

        if label_encoders and "qualification" in label_encoders:
            qual_encoded = int(label_encoders["qualification"].transform([qualification])[0])
        else:
            qual_encoded = 0

        X = np.array([[age, sex_encoded, experience, code_encoded, qual_encoded]], dtype=float)

        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X

        pred = model.predict(X_scaled)
        # ensure we can index into prediction
        pred_value = float(pred[0]) if hasattr(pred, "__len__") else float(pred)

        st.success(f"💰 Estimated Salary for {name or 'Employee'}: ₹{pred_value:,.2f}")

    except Exception as e:
        st.error("❌ Error during prediction:")
        st.code(str(e))
