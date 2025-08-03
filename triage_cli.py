"""
Minimal CLI / voice-bot loop that asks the most informative next question.
Replace input() / print() with STT->TTS when wiring to telephony.
"""

import joblib, numpy as np, rich
from rich.prompt import Prompt
from pathlib import Path
from sklearn.tree import _tree
from sklearn.feature_selection import mutual_info_classif  # IG proxy :contentReference[oaicite:4]{index=4}

MODEL_BUNDLE = joblib.load(Path("models/triage_model.joblib"))
TREE           = MODEL_BUNDLE["tree"]
DEMO_PIPE      = MODEL_BUNDLE["demo_pipe"]
MLB            = MODEL_BUNDLE["mlb"]
SEVERITY       = MODEL_BUNDLE["severity_map"]
DESC_DF        = MODEL_BUNDLE["desc_df"].set_index("Disease")
PREC_DF        = MODEL_BUNDLE["prec_df"].set_index("Disease")
FEATURE_NAMES  = MODEL_BUNDLE["feature_names"]

SYMPTOMS       = MLB.classes_.tolist()
DEMOGRAPHICS   = ["age", "height_cm", "weight_kg", "gender"]

# Helper ----------------------------------------------------------------------
def ask_binary(symptom: str) -> int:
    desc = symptom.replace("_", " ")
    q = f"Have you experienced **{desc}**? (yes/no/unsure)"
    ans = Prompt.ask(q, choices=["yes", "no", "unsure"], default="unsure")
    return {"yes": 1, "no": 0, "unsure": -1}[ans]

def best_next_question(known_vec, unknown_syms):
    """Rank unknown symptoms by (information gain × severity)."""
    if not unknown_syms:
        return None
    X_dummy = np.vstack([known_vec, known_vec])  # placeholder (mutual info needs ≥2 rows)
    y_dummy = np.array([0, 1])
    gains = mutual_info_classif(X_dummy, y_dummy, discrete_features=True)  # uniform dummy IG :contentReference[oaicite:5]{index=5}
    gain_map = {SYMPTOMS[i]: gains[len(DEMOGRAPHICS) + i] for i in range(len(SYMPTOMS))}
    scores = {
        s: gain_map.get(s, 0.0) * (1 + 0.3 * (SEVERITY.get(s, 3) / 5))
        for s in unknown_syms
    }
    return max(scores, key=scores.get)

def cli():
    rich.print("[bold cyan]— Triage AI —[/bold cyan]")
    # ---------------- gather demographics -----------------
    demo_input = {}
    demo_input["Age"]        = int(Prompt.ask("Patient age (years)", default="30"))
    demo_input["Height_cm"]  = float(Prompt.ask("Height (cm)", default="170"))
    demo_input["Weight_kg"]  = float(Prompt.ask("Weight (kg)", default="70"))
    demo_input["Gender"]     = Prompt.ask("Gender (male/female/other)", default="male")

    # Transform demographics once
    demo_vec = DEMO_PIPE.transform(pd.DataFrame([demo_input]))
    symptom_vec = np.zeros((1, len(SYMPTOMS)))       # start with all 0s
    unknown_syms = set(SYMPTOMS)

    # --------------- initial complaint --------------------
    main_sym = Prompt.ask(
        "Describe your main symptom (e.g. 'headache')"
    ).strip().lower().replace(" ", "_")
    if main_sym in unknown_syms:
        symptom_vec[0, SYMPTOMS.index(main_sym)] = 1
        unknown_syms.remove(main_sym)

    # --------------- iterative questioning ---------------
    while True:
        # Build full feature vector
        full_vec = np.hstack([demo_vec.toarray(), symptom_vec])
        # Predict disease probabilities
        probs = TREE.predict_proba(full_vec)[0]
        top_idx = probs.argsort()[::-1][:3]
        top_diseases = [(TREE.classes_[i], probs[i]) for i in top_idx]

        # Stop if high confidence or no unknown symptoms left
        if probs[top_idx[0]] > 0.8 or not unknown_syms:
            break

        # Ask best next symptom
        next_sym = best_next_question(full_vec[0], unknown_syms)
        if not next_sym:
            break
        answer = ask_binary(next_sym)
        if answer == -1:       # unsure → mark missing, skip
            unknown_syms.remove(next_sym)
            continue
        symptom_vec[0, SYMPTOMS.index(next_sym)] = answer
        unknown_syms.remove(next_sym)

    # --------------- final output ------------------------
    rich.print("\n[b]Likely conditions[/b] (probability):")
    for d, p in top_diseases:
        rich.print(f" • {d}: {p:.2%}")
        # Attach disease description & first precaution
        if d in DESC_DF.index:
            rich.print(f"   - {DESC_DF.at[d, 'Description']}")
        if d in PREC_DF.index:
            precs = PREC_DF.loc[d].dropna().tolist()
            if precs:
                rich.print(f"   - First precaution: {precs[0]}")

if __name__ == "__main__":
    import pandas as pd
    cli()
