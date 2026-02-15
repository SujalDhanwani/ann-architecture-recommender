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


def evaluate_model(model, X, y, problem_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(X_tensor)

    metrics = {}

    # ---------------- Regression ----------------
    if problem_type == "regression":
        y_true = y.values
        y_pred = outputs.cpu().numpy().flatten()

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)

        metrics["MSE"] = mse
        metrics["RMSE"] = rmse
        metrics["MAE"] = mae

    # ---------------- Binary Classification ----------------
    elif problem_type == "binary_classification":
        y_true = y.values
        y_pred = torch.sigmoid(outputs).cpu().numpy()
        y_pred = (y_pred >= 0.5).astype(int)

        metrics["Accuracy"] = accuracy_score(y_true, y_pred)
        metrics["Precision"] = precision_score(y_true, y_pred)
        metrics["Recall"] = recall_score(y_true, y_pred)
        metrics["F1"] = f1_score(y_true, y_pred)
        metrics["Confusion_Matrix"] = confusion_matrix(y_true, y_pred)

    # ---------------- Multi-class ----------------
    else:
        y_true = y.values
        y_pred = outputs.argmax(1).cpu().numpy()

        metrics["Accuracy"] = accuracy_score(y_true, y_pred)
        metrics["Macro_F1"] = f1_score(y_true, y_pred, average="macro")

    return metrics
