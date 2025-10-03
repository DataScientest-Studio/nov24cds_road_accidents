import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
import json

print("\n⚡ Running SHAP suite with force-aligned samples...")

# ========== Setup ==========
model = search.best_estimator_
model_name = "BalancedRandomForest_2class_FULL_DATA"
safe_name = model_name.lower().replace(" ", "_")
shap_dir = f"../reports/shap/{safe_name}"
os.makedirs(shap_dir, exist_ok=True)

# ========== Extract pipeline ==========
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

# ========== Get predictions ==========
threshold = best["threshold"]
probas = model.predict_proba(X_test)[:, 1]
y_pred_thr = (probas >= threshold).astype(int)

# ========== Select force plot cases ==========
idx_fn   = np.where((y_test == 1) & (y_pred_thr == 0))[0][:3]
idx_fp   = np.where((y_test == 0) & (y_pred_thr == 1))[0][:3]
idx_tp   = np.where((y_test == 1) & (y_pred_thr == 1))[0][:3]
idx_risk = np.argsort(probas)[-3:][::-1]

force_indices = np.unique(np.concatenate([idx_fn, idx_fp, idx_tp, idx_risk]))
X_sample_raw = X_test.iloc[force_indices].copy()
X_sample_numeric = preprocessor.transform(X_sample_raw)
if hasattr(X_sample_numeric, "toarray"):
    X_sample_numeric = X_sample_numeric.toarray()

# ========== SHAP background ==========
X_bg_raw = X_train.sample(n=1000, random_state=42)
X_bg_transformed = preprocessor.transform(X_bg_raw)
if hasattr(X_bg_transformed, "toarray"):
    X_bg_transformed = X_bg_transformed.toarray()
X_bg = shap.kmeans(X_bg_transformed, 25)

# ========== SHAP explainer ==========
explainer = shap.KernelExplainer(classifier.predict_proba, X_bg)
shap_values = explainer.shap_values(X_sample_numeric)
shap_vals_class1 = shap_values[:, :, 1]

# ========== Feature names ==========
feature_names = preprocessor.get_feature_names_out()

# ========== Summary plot ==========
summary_path = f"{shap_dir}/shap_summary_plot.png"
shap.summary_plot(shap_vals_class1, X_sample_numeric, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig(summary_path, dpi=200)
plt.close()
print(f"✅ Saved summary plot: {summary_path}")

# ========== Bar plot ==========
bar_path = f"{shap_dir}/shap_bar_plot.png"
bar_explanation = shap.Explanation(
    values=shap_vals_class1,
    base_values=explainer.expected_value[1],
    data=X_sample_numeric,
    feature_names=feature_names
)
shap.plots.bar(bar_explanation, show=False)
plt.tight_layout()
plt.savefig(bar_path, dpi=200)
plt.close()
print(f"✅ Saved bar plot: {bar_path}")

# ========== Violin plot ==========
violin_path = f"{shap_dir}/shap_violin_plot.png"
shap.summary_plot(shap_vals_class1, X_sample_numeric, feature_names=feature_names, plot_type="violin", show=False)
plt.tight_layout()
plt.savefig(violin_path, dpi=200)
plt.close()
print(f"✅ Saved violin plot: {violin_path}")

# ========== Dependence + Interaction plot ==========
top_feat_idx = np.argsort(np.abs(shap_vals_class1).mean(0))[-1]
top_feat = feature_names[top_feat_idx]

dep_path = f"{shap_dir}/shap_dependence_{top_feat}.png"
shap.dependence_plot(top_feat, shap_vals_class1, X_sample_numeric, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig(dep_path, dpi=200)
plt.close()
print(f"✅ Saved dependence plot: {dep_path}")

interaction_path = f"{shap_dir}/shap_interaction_{top_feat}.png"
shap.dependence_plot(top_feat, shap_vals_class1, X_sample_numeric, interaction_index='auto', feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig(interaction_path, dpi=200)
plt.close()
print(f"✅ Saved interaction plot: {interaction_path}")

# ========== Heatmap ==========
heatmap_path = f"{shap_dir}/shap_heatmap.png"
plt.figure(figsize=(12, 8))
sns.heatmap(shap_vals_class1, cmap="coolwarm", center=0)
plt.title("SHAP Value Heatmap (Class 1)")
plt.xlabel("Features")
plt.ylabel("Samples")
plt.tight_layout()
plt.savefig(heatmap_path, dpi=200)
plt.close()
print(f"✅ Saved heatmap: {heatmap_path}")

# ========== Force plots & JSON summary ==========
cases = {
    "false_negatives": idx_fn,
    "false_positives": idx_fp,
    "true_positives": idx_tp,
    "high_risk": idx_risk
}

shap_summaries = []
X_test_reset = X_test.reset_index(drop=True)
y_test_reset = pd.Series(y_test).reset_index(drop=True)
probas_reset = pd.Series(probas).reset_index(drop=True)

# Map global index → shap index
shap_idx_map = {idx: i for i, idx in enumerate(force_indices)}

for case_type, indices in cases.items():
    print(f"🔍 SHAP force plots for {case_type.replace('_', ' ')}...")
    for i in indices:
        shap_idx = shap_idx_map.get(i, None)
        if shap_idx is None:
            continue

        base = f"{shap_dir}/shap_force_{case_type}_idx{i}"
        shap.plots.force(
            explainer.expected_value[1],
            shap_vals_class1[shap_idx],
            X_sample_numeric[shap_idx],
            matplotlib=True,
            feature_names=feature_names,
            show=False
        )
        plt.tight_layout()
        plt.savefig(base + ".png", dpi=200)
        plt.close()

        shap_summaries.append({
            "index": int(i),
            "case_type": case_type,
            "y_true": int(y_test_reset[i]),
            "predicted": int(y_pred_thr[i]),
            "proba_positive": float(probas_reset[i]),
            "explanation": [
                {"feature": str(feature_names[j]), "shap_value": float(val)}
                for j, val in enumerate(shap_vals_class1[shap_idx])
                if abs(val) > 0.01
            ]
        })

# ========== Save JSON summary ==========
json_path = f"{shap_dir}/shap_streamlit_summary.json"
with open(json_path, "w") as f:
    json.dump(shap_summaries, f, indent=2)

print(f"\n✅ All SHAP visualizations and JSON summary saved to: {shap_dir}")
