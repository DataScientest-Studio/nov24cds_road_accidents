import warnings
warnings.filterwarnings('ignore')

# ========== Imports ==========
import os, json, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)

from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline

# ========== Load Data ==========
file_path = "../data/processed/df_for_ml.csv"
df = pd.read_csv(file_path, low_memory=False)

# ========== Preprocessing ==========
category_columns = [
    'lum', 'int', 'atm', 'col', 'circ', 'vosp', 'prof', 'plan', 'surf',
    'infra', 'situ', 'senc', 'obs', 'obsm', 'choc', 'catu', 'grav', 'sexe',
    'dayofweek', 'belt_status', 'agg_catr', 'catv_group', 'rush_hour',
    'season', 'age_bin', 'belt_user_type_code'
]
for col in category_columns:
    if col in df.columns:
        df[col] = df[col].astype('category')

for col in ['nbv', 'age']:
    if col in df.columns:
        df[col] = df[col].astype('Int64')

if 'hour' in df.columns:
    df['hour'] = df['hour'].astype('int64')

# ========== Relabel for 2-class ==========
df["grav_label"] = df["grav"].map({1: "Minor/None", 2: "Hospitalized/Killed", 3: "Hospitalized/Killed", 4: "Minor/None"})

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["grav_label"])
class_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Class mapping:", class_mapping)

X = df.drop(columns=["grav", "grav_label"])

# ========== Split ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ========== Preprocessor ==========
cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
num_features = X.select_dtypes(include=['int64', 'float64', 'Int64']).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num_features),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ("vt", VarianceThreshold(threshold=0.01))
    ]), cat_features)
])

# ========== Model Setup ==========
model_name = "BalancedRandomForest_2class_FULL_DATA"
safe_name = model_name.lower().replace(" ", "_")

brf = BalancedRandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

pipeline = ImbPipeline([
    ("preprocessor", preprocessor),
    ("classifier", brf)
])

param_grid = {
    "classifier__max_depth": [10, 20, None],
    "classifier__min_samples_split": [2, 5]
}

# ========== Grid Search ==========
print(f"\n🔍 Grid searching {model_name} ...")
search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    scoring='f1_macro',
    cv=3,
    n_jobs=1,
    verbose=1
)
search.fit(X_train, y_train)
y_pred = search.predict(X_test)
probas = search.predict_proba(X_test)[:, 1]

print(f"✅ Best params: {search.best_params_}")
print(classification_report(y_test, y_pred))

# ========== Threshold Tuning ==========
thresholds = np.linspace(0.05, 0.95, 19)
best = {"threshold": 0.5, "recall": 0, "precision": 0, "f1": 0}

for t in thresholds:
    y_thr = (probas >= t).astype(int)
    prec = precision_score(y_test, y_thr)
    rec = recall_score(y_test, y_thr)
    f1 = f1_score(y_test, y_thr)
    if rec >= 0.70 and f1 > best["f1"]:
        best.update({"threshold": float(t), "recall": rec, "precision": prec, "f1": f1})

print(f"\n🎯 Best threshold: {best['threshold']:.2f} | Recall: {best['recall']:.2f}, F1: {best['f1']:.2f}")

# Final predictions
y_final = (probas >= best["threshold"]).astype(int)

# ========== Confusion Matrices ==========
labels = ["Hospitalized/Killed", "Minor/None"]
cm = confusion_matrix(y_test, y_final)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

os.makedirs("../reports/figures", exist_ok=True)
os.makedirs("../models", exist_ok=True)

# Raw confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title(f"Confusion Matrix - {model_name}")
plt.tight_layout()
plt.savefig(f"../reports/figures/{model_name}_confusion_matrix.png")
plt.close()

# Normalized confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Oranges", xticklabels=labels, yticklabels=labels)
plt.title(f"Confusion Matrix (%) - {model_name}")
plt.tight_layout()
plt.savefig(f"../reports/figures/{model_name}_confusion_matrix_percent.png")
plt.close()

# ========== Save Results ==========
results = {
    "model_name": model_name,
    "best_params": search.best_params_,
    "best_threshold": best,
    "classification_report": classification_report(y_test, y_final, output_dict=True),
    "confusion_matrix": cm.tolist(),
    "confusion_matrix_percent": cm_pct.tolist()
}

# Save model and results
joblib.dump(search.best_estimator_, f"../models/model_{model_name}.joblib")
with open(f"../models/results_{model_name}.json", "w") as f:
    json.dump(results, f, indent=2)

# Save for Streamlit
streamlit_data = {
    model_name: {
        "confusion_matrix": results["confusion_matrix"],
        "confusion_matrix_percent": results["confusion_matrix_percent"]
    }
}
with open(f"../models/streamlit_confusion_matrices_{safe_name}.json", "w") as f:
    json.dump(streamlit_data, f)

print(f"\n✅ All done. Model, results, and plots saved to ../models and ../reports/figures/")
