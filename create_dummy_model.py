import os
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression

# make base directory robust when __file__ may be undefined
try:
    base_dir = os.path.dirname(__file__)
except NameError:
    base_dir = os.getcwd()
MODEL_PATH = os.path.join(base_dir, "linearmodel.pkl")

# Define categories consistent with sala.py defaults
sex_cats = ["Male", "Female"]
lang_cats = ["Python", "Java", "C++", "JavaScript"]
qual_cats = ["Bachelors", "Masters", "PhD"]

# Create label encoders and fit
le_sex = LabelEncoder(); le_sex.fit(sex_cats)
le_lang = LabelEncoder(); le_lang.fit(lang_cats)
le_qual = LabelEncoder(); le_qual.fit(qual_cats)

label_encoders = {"sex": le_sex, "coding_language": le_lang, "qualification": le_qual}

# Generate synthetic training data
rng = np.random.RandomState(42)
n = 1000
age = rng.randint(20, 61, size=n)
sex = rng.choice(sex_cats, size=n)
exp = rng.randint(0, 31, size=n)
lang = rng.choice(lang_cats, size=n)
qual = rng.choice(qual_cats, size=n)

sex_enc = le_sex.transform(sex)
lang_enc = le_lang.transform(lang)
qual_enc = le_qual.transform(qual)

X = np.column_stack([age, sex_enc, exp, lang_enc, qual_enc]).astype(float)

# True underlying formula + noise (matches heuristic in sala.py for reasonable predictions)
y = 20000 + age * 300 + exp * 1500 + lang_enc * 800 + qual_enc * 1200 + rng.normal(0, 1000, size=n)

# Fit scaler and model
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
model = LinearRegression().fit(X_scaled, y)

# Save as dict expected by sala.py
payload = {"model": model, "scaler": scaler, "label_encoders": label_encoders}
joblib.dump(payload, MODEL_PATH)
print(f"Dummy model saved to: {MODEL_PATH}")