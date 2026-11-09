import json
from itertools import chain
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from src.battle_state import create_species_dict
from src.features import build_features
from src.models import (
    RANDOM_STATE,
    build_xgb,
    categorical_columns,
    load_xgb_params,
)


PATH = Path(__file__).resolve().parents[1]
FOLDS = 5
FINAL_SEEDS = 5


def iter_jsonl(file_path):
    with open(file_path, encoding="utf-8") as file:
        for line in file:
            yield json.loads(line)


def load_training_dataset(train_path, test_path, type_chart_path):

    type_chart = pd.read_csv(type_chart_path)
    species = create_species_dict(chain(iter_jsonl(train_path), iter_jsonl(test_path)))
    features = build_features(iter_jsonl(train_path), type_chart, species)
    
    target = features.pop("player_won").astype("int8")
    features.pop("battle_id")
    return features, target, species


def fold_accuracies(target, probabilities, splits):
    accuracies = []
    for _, valid_indices in splits:
        actual = target.iloc[valid_indices]
        predicted = probabilities[valid_indices] >= 0.5
        accuracies.append(accuracy_score(actual, predicted))
    return np.array(accuracies)


def out_of_fold(make_model, frame, target, splits):
    probabilities = np.zeros(len(target))
    for train_indices, valid_indices in splits:
        model = make_model()
        model.fit(frame.iloc[train_indices], target.iloc[train_indices])
        probabilities[valid_indices] = model.predict_proba(frame.iloc[valid_indices])[:, 1]
    return probabilities


def main():
    train_path = PATH/"data"/"raw"/"train.jsonl"
    test_path = PATH/"data"/"raw"/"test.jsonl"
    type_chart_path = PATH/"data"/"raw"/"type_chart_gen2.csv"
    model_path = PATH/"models"/"xgb_battle_model.joblib"
    params_path = PATH/"models"/"best_params.json"
    features, target, species = load_training_dataset(train_path, test_path, type_chart_path)

    feature_columns = features.columns.tolist()
    categorical = categorical_columns(features)
    xgb_params = load_xgb_params(params_path)

    splits = list(StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE).split(features, target))
    xgb_probabilities = out_of_fold(lambda: build_xgb(xgb_params, categorical),features,target,splits)
    xgb_scores = fold_accuracies(target, xgb_probabilities, splits)
    print(f"XGBoost{'':<23} CV accuracy: "f"{xgb_scores.mean():.5f} +/- {xgb_scores.std(ddof=1):.5f}")

    xgb_models = []
    for offset in range(FINAL_SEEDS):
        model = build_xgb(xgb_params, categorical, seed=RANDOM_STATE + offset)
        model.fit(features, target)
        xgb_models.append(model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "xgb_models": xgb_models,
            "feature_columns": feature_columns,
            "species": species,
            "xgb_params": xgb_params,
            "validation_accuracy": xgb_scores.tolist(),
            "validation_accuracy_mean": float(xgb_scores.mean()),
            "validation_accuracy_std": float(xgb_scores.std(ddof=1)),
            "random_state": RANDOM_STATE,
            "folds": FOLDS,
        },
        model_path,
    )
    print(f"CV accuracy {xgb_scores.mean():.5f}")

if __name__ == "__main__":
    main()
