"""
Enhanced training with feature engineering, model comparison, and validation
"""

from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from preprocessor import load_raw, clean_dataset, build_enhanced_feature_matrix

DATA_DIR   = Path("data")          # put the four CSVs here
MODEL_DIR  = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# 1) ---------- ingest & clean -----------------------------------------------
df, sev_map, desc_df, prec_df = load_raw(DATA_DIR)
df = clean_dataset(df)

# 2) ---------- enhanced feature matrix with severity weighting --------------
X, y, demo_pipe, mlb, feat_names, feature_weights = build_enhanced_feature_matrix(df, sev_map)

print(f"Dataset shape: {X.shape}, Classes: {len(np.unique(y))}")
print(f"Feature distribution: Demographics: {len([f for f in feat_names if any(d in f for d in ['age', 'height', 'weight', 'gender'])])}, Symptoms: {len(mlb.classes_)}")

# 3) ---------- train/test split for proper validation -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4) ---------- feature selection for better performance ---------------------
selector = SelectKBest(score_func=mutual_info_classif, k=min(50, X.shape[1]))
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
selected_features = [feat_names[i] for i in selector.get_support(indices=True)]

print(f"Selected {len(selected_features)} most informative features")

# 5) ---------- model comparison and selection --------------------------------
models = {
    'Decision Tree': DecisionTreeClassifier(
        criterion="entropy",
        class_weight="balanced",
        min_samples_leaf=max(2, int(0.01 * len(X_train))),
        max_depth=15,
        random_state=42,
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        criterion="entropy",
        class_weight="balanced",
        min_samples_leaf=max(2, int(0.01 * len(X_train))),
        max_depth=15,
        random_state=42,
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
    )
}

best_model = None
best_score = 0
cv_results = {}

print("\n" + "="*60)
print("MODEL COMPARISON & VALIDATION")
print("="*60)

for name, model in models.items():
    # Cross-validation
    cv_scores = cross_val_score(
        model, X_train_selected, y_train, 
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='balanced_accuracy', 
        n_jobs=-1
    )
    
    cv_results[name] = {
        'mean': cv_scores.mean(),
        'std': cv_scores.std(),
        'scores': cv_scores
    }
    
    print(f"{name}:")
    print(f"  CV Balanced Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
    
    if cv_scores.mean() > best_score:
        best_score = cv_scores.mean()
        best_model = model

# 6) ---------- hyperparameter tuning for best model ------------------------
print(f"\nHyperparameter tuning for best model: {type(best_model).__name__}")

if isinstance(best_model, DecisionTreeClassifier):
    param_grid = {
        "ccp_alpha": [0, 1e-4, 1e-3, 1e-2],
        "max_depth": [10, 15, 20, None],
        "min_samples_split": [2, 5, 10]
    }
elif isinstance(best_model, RandomForestClassifier):
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }
else:  # GradientBoostingClassifier
    param_grid = {
        "n_estimators": [50, 100, 150],
        "learning_rate": [0.05, 0.1, 0.15],
        "max_depth": [3, 6, 9]
    }

gs = GridSearchCV(
    estimator=best_model,
    param_grid=param_grid,
    scoring="balanced_accuracy",
    cv=5,
    n_jobs=-1,
    verbose=1
)

gs.fit(X_train_selected, y_train)
final_model = gs.best_estimator_

# 7) ---------- final evaluation ---------------------------------------------
y_pred = final_model.predict(X_test_selected)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"\nFINAL MODEL PERFORMANCE:")
print(f"Best parameters: {gs.best_params_}")
print(f"CV Best Score: {gs.best_score_:.3f}")
print(f"Test Accuracy: {test_accuracy:.3f}")

if hasattr(final_model, 'get_depth'):
    print(f"Tree Depth: {final_model.get_depth()}")
elif hasattr(final_model, 'estimators_'):
    # Handle different ensemble model types
    if hasattr(final_model, 'estimators_') and len(final_model.estimators_.shape) > 1:
        # GradientBoostingClassifier case
        depths = [tree.get_depth() for tree in final_model.estimators_[:5, 0]]
        print(f"Average Tree Depth (first 5): {np.mean(depths):.1f}")
    else:
        # RandomForestClassifier case  
        depths = [tree.get_depth() for tree in final_model.estimators_[:5]]
        print(f"Average Tree Depth (first 5): {np.mean(depths):.1f}")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8) ---------- persist enhanced model bundle --------------------------------
model_bundle = {
    "model": final_model,
    "demo_pipe": demo_pipe,
    "mlb": mlb,
    "feature_names": feat_names,
    "selected_features": selected_features,
    "feature_selector": selector,
    "feature_weights": feature_weights,
    "severity_map": sev_map,
    "desc_df": desc_df,
    "prec_df": prec_df,
    "cv_results": cv_results,
    "model_type": type(final_model).__name__,
    "best_params": gs.best_params_,
    "test_accuracy": test_accuracy
}

joblib.dump(model_bundle, MODEL_DIR / "triage_model_enhanced.joblib")
print("✔️  Enhanced model saved to models/triage_model_enhanced.joblib")

# Also save the original format for backward compatibility
legacy_bundle = {
    "tree": final_model,
    "demo_pipe": demo_pipe,
    "mlb": mlb,
    "feature_names": feat_names,
    "severity_map": sev_map,
    "desc_df": desc_df,
    "prec_df": prec_df,
}

joblib.dump(legacy_bundle, MODEL_DIR / "triage_model.joblib")
print("✔️  Legacy format saved to models/triage_model.joblib")

print(f"\n🎯 Training Complete! Best model: {type(final_model).__name__}")
print(f"📊 Final test accuracy: {test_accuracy:.1%}")
print(f"🔬 Selected {len(selected_features)} most informative features")
