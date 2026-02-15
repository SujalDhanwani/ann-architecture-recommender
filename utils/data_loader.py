import pandas as pd


def load_csv(path):
    return pd.read_csv(path)



def detect_column_types(df):
    """
    Detect numeric, categorical, boolean, and datetime columns.
    """
    column_types = {
        "numeric": [],
        "categorical": [],
        "boolean": [],
        "datetime": []
    }

    for col in df.columns:
        series = df[col].dropna()
        unique_vals = series.unique()

        # Boolean detection (0/1, True/False, Yes/No)
        bool_like = set(map(str, unique_vals))
        if bool_like.issubset({"0","1","True","False","true","false","YES","NO"}):
            column_types["boolean"].append(col)
            continue

        # Datetime detection
        try:
            pd.to_datetime(series.iloc[0], infer_datetime_format=True)
            column_types["datetime"].append(col)
            continue
        except:
            pass

        # Numeric detection
        if pd.api.types.is_numeric_dtype(df[col]):
            column_types["numeric"].append(col)
            continue

        try:
            series.astype(float)
            column_types["numeric"].append(col)
            continue
        except:
            pass

        # Otherwise categorical
        column_types["categorical"].append(col)

    return column_types


def detect_problem_type_from_all_columns(df):
    """
    If any numeric column has many unique values → regression dataset.
    Otherwise classification.
    """
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    for col in numeric_cols:
        if df[col].nunique() > 20:
            return "regression"

    return "classification"


def detect_target_column(df):
    """
    Detect target column using:
    - regression → highest variance numeric column
    - classification → lowest unique-value column
    """
    problem_type = detect_problem_type_from_all_columns(df)
    n_rows = len(df)
    unique_counts = {col: df[col].nunique() for col in df.columns}

    # Remove ID-like columns (unique == row count)
    filtered_cols = [col for col in df.columns if unique_counts[col] < n_rows]

   
    if problem_type == "classification":
        # smallest unique count = class labels
        min_unique = min(unique_counts[col] for col in filtered_cols)
        candidates = [col for col in filtered_cols if unique_counts[col] == min_unique]
        return candidates[-1]   # choose last candidate

    # ------------------ REGRESSION TARGET ------------------ #
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        variances = df[numeric_cols].var().sort_values(ascending=False)
        return variances.index[0]

    return df.columns[-1]  # fallback


# -------------------------------------------------------------------
# Final Problem Type Detection (using the actual target)
# -------------------------------------------------------------------
def detect_problem_type(df, target_col):
    """
    Final problem type detection using target column.
    """
    series = df[target_col].dropna()
    num_unique = series.nunique()

    # Numeric target
    if pd.api.types.is_numeric_dtype(series):

        if num_unique > 20:
            return "regression"

        if num_unique == 2:
            return "binary_classification"

        return "multi_class_classification"

    # Categorical target
    if num_unique == 2:
        return "binary_classification"

    if num_unique <= 20:
        return "multi_class_classification"

    return "regression"


def select_features(df, target_col):
    """
    Select valid feature columns by removing:
    - target column
    - ID-like columns
    - single-unique columns
    - fully missing columns
    - very high-cardinality categoricals
    """
    
    n_rows = len(df)
    features = []

    for col in df.columns:

        # Remove target itself
        if col == target_col:
            continue

        # Remove fully missing columns
        if df[col].isna().sum() == n_rows:
            continue

        # Remove single-unique columns
        if df[col].nunique() <= 1:
            continue

        # Remove ID-like columns
        if df[col].nunique() == n_rows:
            continue

        # Remove high-cardinality categoricals
        if df[col].dtype == 'object' and df[col].nunique() > 100:
            continue

        # Otherwise keep it
        features.append(col)

    return features


def handle_missing_values(df, column_types):
    """
    Handle missing data using:
    - Drop columns with > 60% missing
    - Numeric -> median
    - Categorical/Boolean/Datetime -> mode
    """

    df = df.copy()
    n_rows = len(df)

    # Drop high-missing columns
    for col in df.columns:
        missing_ratio = df[col].isna().mean()
        if missing_ratio > 0.60:
            df = df.drop(columns=[col])
            continue

    # --- Numeric columns ---
    for col in column_types["numeric"]:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # --- Categorical columns ---
    for col in column_types["categorical"]:
        if col in df.columns:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)

    # --- Boolean columns ---
    for col in column_types["boolean"]:
        if col in df.columns:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)

    # --- Datetime columns ---
    for col in column_types["datetime"]:
        if col in df.columns:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)

    return df



from sklearn.preprocessing import LabelEncoder
import pandas as pd

def encode_categorical_columns(df, column_types):
    df = df.copy()

    # ----- Get low-cardinality & mid-cardinality cols BEFORE modifying df -----
    low_card_cols = [
        col for col in column_types["categorical"]
        if df[col].nunique() <= 20
    ]

    mid_card_cols = [
        col for col in column_types["categorical"]
        if 20 < df[col].nunique() <= 100
    ]

    # ------------------ ONE HOT ENCODE LOW CARDINALITY ------------------
    if len(low_card_cols) > 0:
        df = pd.get_dummies(df, columns=low_card_cols, drop_first=True)

    # ------------------ LABEL ENCODE MID CARDINALITY ------------------
    for col in mid_card_cols:
        if col in df.columns:  # SAFE GUARD
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    return df


def extract_datetime_features(df, column_types):
    """
    Convert datetime columns into numeric features:
    - year, month, day, dayofweek, is_weekend, quarter
    - hour, minute, second (if present)
    """
    df = df.copy()

    for col in column_types["datetime"]:
        # Convert column to datetime format
        df[col] = pd.to_datetime(df[col], errors='coerce')

        # Extract features
        df[col + "_year"] = df[col].dt.year
        df[col + "_month"] = df[col].dt.month
        df[col + "_day"] = df[col].dt.day
        df[col + "_dayofweek"] = df[col].dt.dayofweek
        df[col + "_quarter"] = df[col].dt.quarter
        df[col + "_is_weekend"] = df[col].dt.dayofweek.isin([5, 6]).astype(int)

        # If time exists
        df[col + "_hour"] = df[col].dt.hour
        df[col + "_minute"] = df[col].dt.minute
        df[col + "_second"] = df[col].dt.second

        # Drop original datetime column
        df = df.drop(columns=[col])

    return df


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
import numpy as np

def scale_numeric_columns(df):
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df


def preprocess_dataset(path):

    df = load_csv(path)

    # Detect types
    column_types = detect_column_types(df)

    # Detect target
    target = detect_target_column(df)

    # Select usable features
    features = select_features(df, target)

    # Keep only features + target
    df = df[features + [target]]

    # 🔥 Separate target EARLY
    y = df[target]
    X = df.drop(columns=[target])

    # Re-detect types only on X
    column_types = detect_column_types(X)

    # Missing values
    X = handle_missing_values(X, column_types)

    # Datetime extraction
    X = extract_datetime_features(X, column_types)

    # Re-detect types
    column_types = detect_column_types(X)

    # Encoding
    X = encode_categorical_columns(X, column_types)

    # Re-detect types
    column_types = detect_column_types(X)

    # Scaling
    X = scale_numeric_columns(X)

    # Force numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(0)
    X = X.astype(float)

    return X, y, {
        "target": target,
        "column_types": column_types,
        "features": features,
        "problem_type": detect_problem_type(df, target)
    }
