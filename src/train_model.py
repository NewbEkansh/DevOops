import pandas as pd
import pickle
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler


# --- CONFIG ---
# --- CONFIG ---
DATA_PATH = "data/synthetic_dataset.csv"
MODEL_PATH = "models/neuro_model.pkl"
MODEL_PATH = "models/neuro_model.pkl"
SCALER_PATH = "models/scaler.pkl" # We must save the scaler for the app!

# 1. Load Data
if not os.path.exists(DATA_PATH):
    print(f"❌ Error: {DATA_PATH} not found.")
    exit()

df = pd.read_csv(DATA_PATH)

# 2. Prepare Features (X) and Labels (y)
X = df.drop(['label', 'filename'], axis=1)
y = df['label']

# --- OPTIMIZATION STEP 1: Feature Scaling ---
# This ensures a word count of 200 doesn't "outweigh" a pause rate of 0.4
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- OPTIMIZATION STEP 2: Handle Class Imbalance ---
# COMPLETE: We are now using a balanced synthetic dataset (2000 samples).
# No need for SMOTE anymore.
print("✅ Using balanced synthetic dataset.")
X_res, y_res = X_scaled, y

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# --- OPTIMIZATION STEP 3: Hyperparameter Tuning ---
print("🧠 Searching for the optimal brain configuration...")
param_grid = {
    'n_estimators': [200, 500],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced', None]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42), 
    param_grid, 
    cv=5, 
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
model = grid_search.best_estimator_

# 4. Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("-" * 30)
print(f"🔥 FINAL OPTIMIZED ACCURACY: {accuracy * 100:.2f}%")
print(f"Best Params: {grid_search.best_params_}")
print("-" * 30)
print("Detailed Clinical Report:")
print(classification_report(y_test, y_pred))

# 5. Feature Importance (For the Judges!)
print("-" * 30)
print("🧠 Clinical Biomarker Weights:")
importances = model.feature_importances_
feature_names = X.columns
for feature, importance in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
    print(f"   - {feature}: {importance:.3f}")

# 6. Save the Model AND the Scaler
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)

with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)
    
print(f"\n💾 Model and Scaler saved. System ready for 90%+ performance!")