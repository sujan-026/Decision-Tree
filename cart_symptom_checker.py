"""Simple CART-based symptom questioner.

This module trains a DecisionTree classifier using the existing dataset
and exposes a small CLI that interactively asks the user about symptoms.
The next question is chosen by following the structure of the trained
decision tree: at each node we ask about the symptom used for splitting
and traverse left/right depending on the user's answer.
"""

from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree

from preprocessor import load_raw, clean_dataset, build_feature_matrix

MODEL_PATH = Path("models/cart_model.joblib")


def train_cart_model(data_dir: Path = Path("data")) -> None:
    """Train a single CART decision tree and persist it.

    The function uses the basic feature matrix (symptoms + demographics)
    so that the resulting model remains easy to interpret and traverse.
    """
    df, sev_map, desc_df, prec_df = load_raw(data_dir)
    df = clean_dataset(df)
    X, y, demo_pipe, mlb, feature_names = build_feature_matrix(df)

    clf = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=15,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X, y)

    bundle = {
        "tree": clf,
        "demo_pipe": demo_pipe,
        "mlb": mlb,
        "feature_names": feature_names,
        "severity_map": sev_map,
        "desc_df": desc_df,
        "prec_df": prec_df,
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)


def load_model():
    """Load the CART model, training it if necessary."""
    if not MODEL_PATH.exists():
        train_cart_model()
    return joblib.load(MODEL_PATH)


def interactive_session():
    """Run an interactive symptom checking session."""
    bundle = load_model()
    tree_struct = bundle["tree"].tree_
    feature_names = bundle["feature_names"]
    mlb = bundle["mlb"]
    desc_df = bundle["desc_df"].set_index("Disease")
    prec_df = bundle["prec_df"].set_index("Disease")

    symptoms = mlb.classes_.tolist()

    # gather demographics first
    print("--- Patient information ---")
    demo = {
        "Age": int(input("Age (years): ") or 30),
        "Height_cm": float(input("Height (cm): ") or 170),
        "Weight_kg": float(input("Weight (kg): ") or 70),
        "Gender": (input("Gender (male/female/other): ") or "male"),
    }
    demo_vec = bundle["demo_pipe"].transform(pd.DataFrame([demo]))
    x = np.hstack([demo_vec.toarray(), np.zeros(len(symptoms))])
    unknown = set(symptoms)

    print("\nDescribe your main symptom")
    main = input("Main symptom: ").strip().lower().replace(" ", "_")
    if main in unknown:
        idx = symptoms.index(main)
        x[0, len(bundle["demo_pipe"].get_feature_names_out()) + idx] = 1
        unknown.remove(main)

    # traverse tree, asking about unknown symptoms as encountered
    node = 0
    while tree_struct.feature[node] != _tree.TREE_UNDEFINED:
        feat_idx = tree_struct.feature[node]
        name = feature_names[feat_idx]
        thresh = tree_struct.threshold[node]
        if name in unknown:
            ans = input(f"Do you have {name.replace('_', ' ')}? (y/n): ").strip().lower()
            val = 1 if ans.startswith("y") else 0
            x[0, feat_idx] = val
            unknown.remove(name)
        val = x[0, feat_idx]
        node = (
            tree_struct.children_left[node] if val <= thresh else tree_struct.children_right[node]
        )

    # final prediction using completed feature vector
    clf = bundle["tree"]
    probs = clf.predict_proba(x)[0]
    top = np.argsort(probs)[::-1][:3]
    print("\nLikely conditions:")
    for idx in top:
        disease = clf.classes_[idx]
        prob = probs[idx]
        print(f" - {disease} ({prob:.1%})")
        if disease in desc_df.index:
            print("   ", desc_df.at[disease, "Description"])
        if disease in prec_df.index:
            first_prec = prec_df.loc[disease].dropna().tolist()
            if first_prec:
                print("   First precaution:", first_prec[0])


if __name__ == "__main__":
    interactive_session()
