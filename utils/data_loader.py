import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


# ==========================================================
# LOAD CSV
# ==========================================================
def load_csv(path):
    return pd.read_csv(path)


# ==========================================================
# DETECT ID-LIKE COLUMNS
# ==========================================================
def detect_id_columns(df):

    id_cols = []

    for col in df.columns:

        # Column name contains ID-like patterns
        if any(key in col.lower() for key in [
            "id", "participant", "index", "uid", "user", "sr", "serial", "record"
        ]):
            id_cols.append(col)
            continue

        # Column is basically row index
        if df[col].nunique() == len(df):
            id_cols.append(col)
            continue

        # Purely increasing 1..N sequence
        if np.array_equal(df[col].values, np.arange(1, len(df) + 1)):
            id_cols.append(col)
            continue

    return id_cols


# ==========================================================
# DETECT COLUMN TYPES
# ==========================================================
def detect_column_types(df):

    types = {"numeric": [], "categorical": [], "boolean": [], "datetime": []}

    for col in df.columns:

        if pd.api.types.is_bool_dtype(df[col]):
            types["boolean"].append(col)
            continue

        if pd.api.types.is_datetime64_any_dtype(df[col]):
            types["datetime"].append(col)
            continue

        # Try datetime parsing for mixed type
        try:
            pd.to_datetime(df[col], errors='raise')
            types["datetime"].append(col)
            continue
        except:
            pass

        if pd.api.types.is_numeric_dtype(df[col]):
            types["numeric"].append(col)
            continue

        types["categorical"].append(col)

    return types


# ==========================================================
# TARGET DETECTION (INTELLIGENT)
# ==========================================================
def detect_target_column(df):

    id_cols = detect_id_columns(df)
    candidates = [c for c in df.columns if c not in id_cols]

    # Priority target names
    priority_names = ["target", "label", "output", "y"]
    for col in candidates:
        if col.lower() in priority_names:
            return col

    # Choose numeric with highest variance
    numeric_cols = df[candidates].select_dtypes(include=['int', 'float']).columns
    if len(numeric_cols) > 0:
        return df[numeric_cols].var().sort_values(ascending=False).index[0]

    # Last fallback
    return candidates[-1]


# ==========================================================
# PROBLEM TYPE (CORRECT)
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

    # categorical
    if uniq == 2:
        return "binary_classification"

    return "multi_class_classification"


# ==========================================================
# INTELLIGENT FEATURE SELECTION
# ==========================================================
def select_features(df, target):

    id_cols = detect_id_columns(df)
    features = []

    for col in df.columns:

        if col == target:
            continue

        if col in id_cols:
            continue

        # fully missing
        if df[col].isna().sum() == len(df):
            continue

        # no variance
        if df[col].nunique() <= 1:
            continue

        # too high cardinality
        if df[col].dtype == "object" and df[col].nunique() > 100:
            continue

        features.append(col)

    return features


# ==========================================================
# MISSING VALUES
# ==========================================================
def handle_missing(df, types):

    df = df.copy()

    for col in types["numeric"]:
        if col in df:
            df[col] = df[col].fillna(df[col].median())

    for col in types["categorical"]:
        if col in df:
            df[col] = df[col].fillna(df[col].mode()[0])

    for col in types["boolean"]:
        if col in df:
            df[col] = df[col].fillna(df[col].mode()[0])

    for col in types["datetime"]:
        if col in df:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


# ==========================================================
# DATETIME EXPANSION
# ==========================================================
def extract_datetime(df, types):
    df = df.copy()

    for col in types["datetime"]:
        df[col] = pd.to_datetime(df[col], errors='coerce')

        df[col + "_year"] = df[col].dt.year
        df[col + "_month"] = df[col].dt.month
        df[col + "_day"] = df[col].dt.day
        df[col + "_dow"] = df[col].dt.dayofweek

        df.drop(columns=[col], inplace=True)

    return df


# ==========================================================
# ENCODING
# ==========================================================
def encode_categorical(df):
    return pd.get_dummies(df, drop_first=True)


# ==========================================================
# SCALE X FEATURES
# ==========================================================
def scale_X(df):

    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Skip binary/dummy columns
    cont_cols = [c for c in numeric_cols if df[c].nunique() > 2]

    # Skip zero variance
    cont_cols = [c for c in cont_cols if df[c].std() > 0]

    if cont_cols:
        scaler = StandardScaler()
        df[cont_cols] = scaler.fit_transform(df[cont_cols])

    return df


# ==========================================================
# MAIN PREPROCESS PIPELINE
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

    # 3️⃣ Handle missing + types
    types = detect_column_types(X)
    X = handle_missing(X, types)

    X = extract_datetime(X, types)
    X = encode_categorical(X)
    X = scale_X(X)

    # Safety cleanup
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)

    # 4️⃣ Target scaling for regression
    y_scaler = None

    if problem_type == "regression":
        y_scaler = StandardScaler()
        y = y_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
        y = pd.Series(y)

    return X, y, {
        "target": target,
        "problem_type": problem_type,
        "features": features,
        "y_scaler": y_scaler
    }
