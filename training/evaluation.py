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


def evaluate_model(
    model,
    X,
    y,
    problem_type,
    y_scaler=None,
    original_y=None
):
    """
    Evaluate trained model.

    For regression:
        - If y_scaler is provided, predictions are inverse transformed
        - original_y must be provided (unscaled target)

    For classification:
        - y should be original labels
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Convert X to tensor
    X_tensor = torch.tensor(
        X.values,
        dtype=torch.float32
    ).to(device)

    with torch.no_grad():
        outputs = model(X_tensor)

    metrics = {}

    # ==========================================================
    # REGRESSION
    # ==========================================================
    if problem_type == "regression":

        y_pred = outputs.cpu().numpy().reshape(-1, 1)

        # Inverse transform predictions if scaler exists
        if y_scaler is not None:
            y_pred = y_scaler.inverse_transform(y_pred)

        # Use original unscaled y
        if original_y is not None:
            y_true = original_y.values.reshape(-1, 1)
        else:
            y_true = y.values.reshape(-1, 1)

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)

        metrics["MSE"] = mse
        metrics["RMSE"] = rmse
        metrics["MAE"] = mae

    # ==========================================================
    # BINARY CLASSIFICATION
    # ==========================================================
    elif problem_type == "binary_classification":

        y_true = y.values.astype(int)

        probs = torch.sigmoid(outputs)
        y_pred = (probs >= 0.5).int().cpu().numpy().flatten()

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

        y_pred = torch.argmax(outputs, dim=1).cpu().numpy()

        metrics["Accuracy"] = accuracy_score(y_true, y_pred)
        metrics["Macro_F1"] = f1_score(y_true, y_pred, average="macro")
        metrics["Weighted_F1"] = f1_score(y_true, y_pred, average="weighted")
        metrics["Confusion_Matrix"] = confusion_matrix(y_true, y_pred)

    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")

    return metrics
