import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# ==========================================================
# LOAD CSV (UTF-8 fallback safe)
# ==========================================================
def load_csv(path):
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1", engine="python")


# ==========================================================
# DETECT ID-LIKE COLUMNS
# ==========================================================
def detect_id_columns(df):
    id_cols = []

    for col in df.columns:

        col_lower = col.lower()

        # Name-based detection
        if any(key in col_lower for key in ["id", "participant", "index", "sr", "uid", "user", "record"]):
            id_cols.append(col)
            continue

        # Unique for every row → ID-like
        if df[col].nunique() == len(df):
            id_cols.append(col)
            continue

    return id_cols


# ==========================================================
# TARGET DETECTION (SMART)
# ==========================================================
def detect_target_column(df):

    id_cols = detect_id_columns(df)
    candidates = [c for c in df.columns if c not in id_cols]

    # Priority names if exist
    priority = ["target", "label", "output", "y"]
    for col in candidates:
        if col.lower() in priority:
            return col

    # Highest variance numeric column
    numeric = df[candidates].select_dtypes(include=[np.number])
    if len(numeric.columns) > 0:
        return numeric.var().sort_values(ascending=False).index[0]

    # Fallback last column
    return candidates[-1]


# ==========================================================
# PROBLEM TYPE
# ==========================================================
def detect_problem_type(df, target):
    y = df[target]
    uniq = y.nunique()

    if pd.api.types.is_numeric_dtype(y):
        if uniq > 20:
            return "regression"
        if uniq == 2:
            return "binary_classification"
        return "multi_class_classification"

    if uniq == 2:
        return "binary_classification"

    return "multi_class_classification"


# ==========================================================
# FEATURE SELECTION (VERY CLEAN)
# ==========================================================
def select_features(df, target):
    id_cols = detect_id_columns(df)
    features = []

    for col in df.columns:

        if col == target:
            continue

        if col in id_cols:
            continue

        # Fully missing
        if df[col].isna().sum() == len(df):
            continue

        # Constant values
        if df[col].nunique() <= 1:
            continue

        # Very high-cardinality categoricals (>100 unique)
        if df[col].dtype == object and df[col].nunique() > 100:
            continue

        features.append(col)

    return features


# ==========================================================
# SAFE SCALING (ONLY NUMERIC, NO CATEGORICAL)
# ==========================================================
def scale_X(X):

    X = X.copy()

    numeric_cols = X.select_dtypes(include=[np.number]).columns

    # Remove binary/dummy & zero variance
    good_cols = [
        col for col in numeric_cols
        if X[col].nunique() > 2 and X[col].std() > 1e-8
    ]

    if good_cols:
        scaler = StandardScaler()
        X[good_cols] = scaler.fit_transform(X[good_cols])

    return X


# ==========================================================
# MAIN PREPROCESS FUNCTION (FINAL)
# ==========================================================
def preprocess_dataset(path):

    df = load_csv(path)

    # 1️⃣ Target detection
    target = detect_target_column(df)
    problem_type = detect_problem_type(df, target)

    # 2️⃣ Feature selection
    features = select_features(df, target)

    X = df[features].copy()
    y = df[target].copy()

    # 3️⃣ Make X fully numeric
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    # 4️⃣ Clip extreme values (fixes exploding gradients)
    X = X.clip(-1e6, 1e6)

    # 5️⃣ Scale X
    X = scale_X(X)
    X = X.astype(float)

    # 6️⃣ Target scaling ONLY for regression
    y_scaler = None
    original_y = None

    if problem_type == "regression":
        y_scaler = StandardScaler()
        original_y = y.values.reshape(-1, 1)
        y = y_scaler.fit_transform(original_y).flatten()
        y = pd.Series(y)

    return X, y, {
        "target": target,
        "problem_type": problem_type,
        "features": features,
        "y_scaler": y_scaler,
        "original_y": original_y
    }