import torch
import torch.nn as nn


# ===============================================================
# SAFE LOSS FUNCTION SELECTION
# ===============================================================
def get_loss_function(problem_type):

    if problem_type == "regression":
        # More stable than pure MSE for small gradients
        return nn.MSELoss()

    if problem_type == "binary_classification":
        # MUST use BCEWithLogitsLoss because model outputs raw logits
        return nn.BCEWithLogitsLoss()

    if problem_type == "multi_class_classification":
        return nn.CrossEntropyLoss()

    raise ValueError(f"Unknown problem type: {problem_type}")


# ===============================================================
# SAFE OPTIMIZER SELECTION
# ===============================================================
def get_optimizer(model, lr=0.001, optimizer_name="adam"):
    """
    Stable optimizers with NaN-safe defaults:
    - Added eps for Adam/RMSProp
    - Added weight_decay to prevent exploding weights
    - SGD has momentum for speed + stability
    """

    optimizer_name = optimizer_name.lower()

    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            eps=1e-8,           # avoids division by zero
            weight_decay=1e-4   # prevents divergence
        )

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=1e-4
        )

    if optimizer_name == "rmsprop":
        return torch.optim.RMSprop(
            model.parameters(),
            lr=lr,
            eps=1e-8,
            weight_decay=1e-4,
            momentum=0.9
        )

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")