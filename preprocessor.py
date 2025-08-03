"""
Enhanced data cleaning & feature engineering for medical-triage decision tree
Author: <you>
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

SYMPTOM_COLS = [f"Symptom_{i}" for i in range(1, 18)]        # 17 slots in dataset.csv
DEMOGRAPHIC_COLS = ["Age", "Height_cm", "Weight_kg", "Gender"]

def load_raw(data_dir: Path):
    df      = pd.read_csv(data_dir / "dataset.csv")
    sev_map = (
        pd.read_csv(data_dir / "Symptom-severity.csv")
        .assign(Symptom=lambda d: d.Symptom.str.strip().str.lower().str.replace(" ", "_"))
        .set_index("Symptom")["weight"]
        .to_dict()
    )
    desc_df  = pd.read_csv(data_dir / "symptom_Description.csv")
    prec_df  = pd.read_csv(data_dir / "symptom_precaution.csv")
    return df, sev_map, desc_df, prec_df

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise text and ensure NaNs for empty symptoms."""
    df = df.copy()
    for c in SYMPTOM_COLS:
        df[c] = (
            df[c]
            .astype("string")
            .str.strip()
            .str.lower()
            .str.replace(" ", "_", regex=False)
            .replace({"nan": pd.NA})
        )
    # Ensure demographic columns exist; add dummies if training set lacks them
    for col in DEMOGRAPHIC_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    return df

def build_feature_matrix(df: pd.DataFrame):
    """Return X (2-D np.array), y (labels), fitted preprocessor, and symptom list."""
    # --- 1) binary symptom matrix -------------------------------------------
    multilabel = df[SYMPTOM_COLS].apply(
        lambda r: [s for s in r if pd.notna(s)], axis=1
    )
    mlb = MultiLabelBinarizer()  # :contentReference[oaicite:0]{index=0}
    X_sym = mlb.fit_transform(multilabel)

    # --- 2) demographics ----------------------------------------------------
    demo_df = df[DEMOGRAPHIC_COLS]
    cat = ["Gender"]
    num = ["Age", "Height_cm", "Weight_kg"]

    pre_demo = ColumnTransformer(
        [
            ("num", StandardScaler(), num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ],
        remainder="drop",
        sparse_threshold=0,
    )
    X_demo = pre_demo.fit_transform(demo_df)

    # --- 3) concatenate -----------------------------------------------------
    import numpy as np

    X = np.hstack([X_demo, X_sym])
    y = df["Disease"]
    feature_names = (
        list(pre_demo.get_feature_names_out()) + mlb.classes_.tolist()
    )
    return X, y, pre_demo, mlb, feature_names


def build_enhanced_feature_matrix(df: pd.DataFrame, severity_map: dict):
    """
    Enhanced feature matrix with severity weighting, interaction features, and better preprocessing.
    Returns X, y, preprocessor, mlb, feature_names, feature_weights
    """
    # --- 1) binary symptom matrix with severity weighting -------------------
    multilabel = df[SYMPTOM_COLS].apply(
        lambda r: [s for s in r if pd.notna(s)], axis=1
    )
    mlb = MultiLabelBinarizer()
    X_sym = mlb.fit_transform(multilabel)
    
    # Apply severity weighting to symptom features
    severity_weights = np.array([severity_map.get(sym, 3.0) for sym in mlb.classes_])
    severity_weights = severity_weights / severity_weights.max()  # Normalize to [0,1]
    X_sym_weighted = X_sym * severity_weights
    
    # --- 2) enhanced demographics with derived features ---------------------
    demo_df = df[DEMOGRAPHIC_COLS].copy()
    
    # Add derived demographic features
    if 'Age' in demo_df.columns and 'Height_cm' in demo_df.columns and 'Weight_kg' in demo_df.columns:
        # Fill missing values with reasonable defaults
        demo_df['Age'] = demo_df['Age'].fillna(30)
        demo_df['Height_cm'] = demo_df['Height_cm'].fillna(170)
        demo_df['Weight_kg'] = demo_df['Weight_kg'].fillna(70)
        
        # BMI calculation
        demo_df['BMI'] = demo_df['Weight_kg'] / ((demo_df['Height_cm'] / 100) ** 2)
        
        # Age groups (with safe handling)
        demo_df['Age_Group'] = pd.cut(
            demo_df['Age'], 
            bins=[0, 18, 30, 50, 70, 120], 
            labels=['Child', 'Young Adult', 'Adult', 'Middle Age', 'Senior'],
            include_lowest=True
        ).astype(str)
        
        # BMI categories (with safe handling)
        demo_df['BMI_Category'] = pd.cut(
            demo_df['BMI'],
            bins=[0, 18.5, 25, 30, 100],
            labels=['Underweight', 'Normal', 'Overweight', 'Obese'],
            include_lowest=True
        ).astype(str)
        
    # Handle Gender column separately (it might exist without other demographic columns)
    if 'Gender' in demo_df.columns:
        demo_df['Gender'] = demo_df['Gender'].fillna('unknown')
    
    # Define categorical and numerical columns
    cat_cols = ['Gender', 'Age_Group', 'BMI_Category']
    num_cols = ['Age', 'Height_cm', 'Weight_kg', 'BMI']
    
    # Filter columns that actually exist
    cat_cols = [col for col in cat_cols if col in demo_df.columns]
    num_cols = [col for col in num_cols if col in demo_df.columns]
    
    pre_demo = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop='first'), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0,
    )
    X_demo = pre_demo.fit_transform(demo_df)
    
    # --- 3) symptom interaction features ------------------------------------
    # Create features for common symptom combinations
    symptom_interactions = []
    common_pairs = [
        ('fever', 'headache'),
        ('cough', 'fever'),
        ('nausea', 'vomiting'),
        ('chest_pain', 'breathlessness'),
        ('abdominal_pain', 'nausea'),
    ]
    
    for sym1, sym2 in common_pairs:
        if sym1 in mlb.classes_ and sym2 in mlb.classes_:
            idx1 = list(mlb.classes_).index(sym1)
            idx2 = list(mlb.classes_).index(sym2)
            interaction = X_sym[:, idx1] * X_sym[:, idx2]
            symptom_interactions.append(interaction.reshape(-1, 1))
    
    if symptom_interactions:
        X_interactions = np.hstack(symptom_interactions)
    else:
        X_interactions = np.empty((X_sym.shape[0], 0))
    
    # --- 4) symptom count and severity features -----------------------------
    symptom_count = X_sym.sum(axis=1).reshape(-1, 1)
    total_severity = (X_sym * severity_weights).sum(axis=1).reshape(-1, 1)
    avg_severity = np.divide(
        total_severity, 
        symptom_count,
        out=np.zeros_like(total_severity), 
        where=symptom_count!=0
    )
    
    severity_features = np.hstack([symptom_count, total_severity, avg_severity])
    
    # --- 5) concatenate all features ---------------------------------------
    X = np.hstack([
        X_demo,
        X_sym_weighted,
        X_interactions,
        severity_features
    ])
    
    y = df["Disease"]
    
    # --- 6) create comprehensive feature names -----------------------------
    demo_feature_names = list(pre_demo.get_feature_names_out())
    symptom_feature_names = [f"symptom_{sym}" for sym in mlb.classes_]
    interaction_feature_names = [f"interaction_{sym1}_{sym2}" for sym1, sym2 in common_pairs 
                               if sym1 in mlb.classes_ and sym2 in mlb.classes_]
    severity_feature_names = ['symptom_count', 'total_severity', 'avg_severity']
    
    feature_names = (
        demo_feature_names + 
        symptom_feature_names + 
        interaction_feature_names + 
        severity_feature_names
    )
    
    # --- 7) create feature importance weights ------------------------------
    feature_weights = np.ones(len(feature_names))
    
    # Higher weight for severity-based features
    severity_start_idx = len(demo_feature_names) + len(symptom_feature_names) + len(interaction_feature_names)
    feature_weights[severity_start_idx:] = 1.5
    
    # Higher weight for interaction features
    if interaction_feature_names:
        interaction_start_idx = len(demo_feature_names) + len(symptom_feature_names)
        interaction_end_idx = interaction_start_idx + len(interaction_feature_names)
        feature_weights[interaction_start_idx:interaction_end_idx] = 1.3
    
    return X, y, pre_demo, mlb, feature_names, feature_weights


def get_symptom_severity_score(symptoms: list, severity_map: dict) -> float:
    """Calculate total severity score for a list of symptoms."""
    return sum(severity_map.get(sym, 3.0) for sym in symptoms)


def get_demographic_risk_factors(age: int, bmi: float, gender: str) -> list:
    """Identify demographic risk factors based on age, BMI, and gender."""
    risk_factors = []
    
    if age >= 65:
        risk_factors.append("elderly_risk")
    elif age <= 5:
        risk_factors.append("pediatric_risk")
    
    if bmi >= 30:
        risk_factors.append("obesity_risk")
    elif bmi < 18.5:
        risk_factors.append("underweight_risk")
    
    if gender.lower() == 'female':
        risk_factors.append("female_specific_conditions")
    
    return risk_factors
