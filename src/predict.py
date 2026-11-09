from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.features import build_features
from src.train import iter_jsonl


PATH = Path(__file__).resolve().parents[1]

def predict_probabilities(saved_model, features):
    model_probabilities = []
    for model in saved_model["xgb_models"]:
        probabilities = model.predict_proba(features)[:, 1]
        model_probabilities.append(probabilities)
    return np.mean(model_probabilities, axis=0)


def main():
    test_path = PATH/"data"/"raw"/"test.jsonl"
    type_chart_path = PATH/ "data"/"raw"/"type_chart_gen2.csv"
    model_path = PATH/"models"/"xgb_battle_model.joblib"
    sample_submission_path = PATH/"data"/"raw"/"sample_submission.csv"
    output_path = PATH/"submissions"/"submission.csv"

    saved_model = joblib.load(model_path)
    type_chart = pd.read_csv(type_chart_path)
    features = build_features(iter_jsonl(test_path), type_chart, saved_model["species"])
    battle_ids = features.pop("battle_id")
    feature_columns = saved_model["feature_columns"]
    features = features.reindex(columns=feature_columns, fill_value=0)
    probabilities = predict_probabilities(saved_model, features)
    predictions = (probabilities >= 0.5).astype("int8")

    sample = pd.read_csv(sample_submission_path)
    predictions_by_id = pd.Series(predictions, index=battle_ids.to_numpy())
    submission = sample.copy()
    submission["player_won"] = submission["battle_id"].map(predictions_by_id)
    submission["player_won"] = submission["player_won"].astype("int8")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()
