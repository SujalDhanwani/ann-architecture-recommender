import torch
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def evaluate_model(model, X, y, problem_type, y_scaler=None):
    """
    SAFE & STABLE MODEL EVALUATION
    --------------------------------
    ✔ Does NOT use original_y (removes old bug)
    ✔ Works for regression / binary / multi-class
    ✔ Automatically inverse-scales regression target
    ✔ Prevents NaN explosions by clipping outputs
    ✔ Ensures shape correctness
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # --------------------------------------------------------
    # Convert X to tensor
    # --------------------------------------------------------
    X_tensor = torch.tensor(X.values, dtype=torch.float32, device=device)

    with torch.no_grad():
        outputs = model(X_tensor)

    metrics = {}

    # ==========================================================
    # REGRESSION
    # ==========================================================
    if problem_type == "regression":

        # Ensure correct shape
        y_pred = outputs.cpu().numpy().reshape(-1, 1)
        y_true = y.values.reshape(-1, 1)

        # Clamp to avoid NaN explosion
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e9, neginf=-1e9)

        # Inverse scale if scaler exists
        if y_scaler is not None:
            try:
                y_pred = y_scaler.inverse_transform(y_pred)
                y_true = y_scaler.inverse_transform(y_true)
            except Exception:
                # Safety fallback
                y_pred = y_pred
                y_true = y_true

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)

        metrics["MSE"] = float(mse)
        metrics["RMSE"] = float(rmse)
        metrics["MAE"] = float(mae)

    # ==========================================================
    # BINARY CLASSIFICATION
    # ==========================================================
    elif problem_type == "binary_classification":

        y_true = y.values.astype(int)

        probs = torch.sigmoid(outputs).cpu().numpy().flatten()
        probs = np.nan_to_num(probs, nan=0.5)  # safe fallback

        y_pred = (probs >= 0.5).astype(int)

        metrics["Accuracy"] = accuracy_score(y_true, y_pred)
        metrics["Precision"] = precision_score(y_true, y_pred, zero_division=0)
        metrics["Recall"] = recall_score(y_true, y_pred, zero_division=0)
        metrics["F1"] = f1_score(y_true, y_pred, zero_division=0)
        metrics["Confusion_Matrix"] = confusion_matrix(y_true, y_pred)

    # ==========================================================
    # MULTI-CLASS CLASSIFICATION
    # ==========================================================
    elif problem_type == "multi_class_classification":

        y_true = y.values.astype(int)

        y_pred = outputs.argmax(1).cpu().numpy()
        y_pred = np.nan_to_num(y_pred, nan=0)

        metrics["Accuracy"] = accuracy_score(y_true, y_pred)
        metrics["Macro_F1"] = f1_score(y_true, y_pred, average="macro")
        metrics["Weighted_F1"] = f1_score(y_true, y_pred, average="weighted")
        metrics["Confusion_Matrix"] = confusion_matrix(y_true, y_pred)

    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")

    return metrics