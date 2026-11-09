import json

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


RANDOM_STATE = 42
DEFAULT_XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.04,
    "min_child_weight": 5,
    "subsample": 0.85,
    "colsample_bynode": 0.4,
    "reg_alpha": 0.0,
    "reg_lambda": 4.0,
    "gamma": 0.0,
}


def categorical_columns(frame):
    categorical = []
    for column, value in frame.iloc[0].items():
        if isinstance(value, str):
            categorical.append(column)
    return categorical


def load_xgb_params(path):
    if path is not None and path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return dict(DEFAULT_XGB_PARAMS)


def build_xgb(params, categorical, seed=RANDOM_STATE):
    """One-hot encode the categories and train XGBoost on a dense matrix.

    The output stays dense on purpose: XGBoost reads the zeros left out of a
    sparse matrix as missing values, and here zero is a real measurement.
    Encoding is fitted inside the pipeline, so each fold uses only its own
    training data.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=5,
                    sparse_output=False,
                    dtype=np.float32,
                ),
                categorical,
            )
        ],
        remainder="passthrough",
        sparse_threshold=0.0,
    )
    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
        **params,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])
