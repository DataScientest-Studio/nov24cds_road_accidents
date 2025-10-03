import argparse, os, json, joblib
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Run predictions with saved BalancedRandomForest pipeline.")
    parser.add_argument('--input', '-i', type=str, default='../data/processed/df_for_ml.csv',
                        help='Path to input CSV with same schema as training data.')
    parser.add_argument('--output', '-o', type=str, default='../reports/predictions.csv',
                        help='Path to save predictions CSV.')
    parser.add_argument('--model', '-m', type=str, default='../models/model_BalancedRandomForest_2class_FULL_DATA.joblib',
                        help='Path to saved joblib model.')
    parser.add_argument('--results', '-r', type=str, default='../models/results_BalancedRandomForest_2class_FULL_DATA.json',
                        help='Path to results JSON (for threshold).')
    args = parser.parse_args()

    # Load pipeline
    model = joblib.load(args.model)

    # Load threshold from results JSON (fallback to 0.5)
    threshold = 0.5
    if os.path.exists(args.results):
        with open(args.results, 'r') as f:
            res = json.load(f)
            threshold = float(res.get('best_threshold', {}).get('threshold', 0.5))

    # Load data
    df = pd.read_csv(args.input, low_memory=False)

    # If the original training code encoded grav into a label, drop it for inference if present
    for col in ['grav']:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Predict probabilities
    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(df)[:, 1]
        preds = (probas >= threshold).astype(int)
    else:
        preds = model.predict(df)
        probas = np.full_like(preds, np.nan, dtype=float)

    out = df.copy()
    out['proba_positive'] = probas
    out['prediction'] = preds

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"✅ Saved predictions to {args.output} using threshold={threshold:.2f}")

if __name__ == '__main__':
    main()
