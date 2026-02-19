import torch
import torch.nn as nn


# ==========================================================
# LOSS FUNCTION FACTORY
# ==========================================================
def get_loss_function(problem_type):
    """
    Returns appropriate loss function based on problem type.
    """

    if problem_type == "regression":
        return nn.MSELoss()

    elif problem_type == "binary_classification":
        # Uses logits → do NOT apply sigmoid in model
        return nn.BCEWithLogitsLoss()

    elif problem_type == "multi_class_classification":
        # Uses raw logits → do NOT apply softmax in model
        return nn.CrossEntropyLoss()

    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")


# ==========================================================
# OPTIMIZER FACTORY
# ==========================================================
def get_optimizer(model, lr=0.001, optimizer_name="adam"):
    """
    Returns optimizer based on name.
    """

    optimizer_name = optimizer_name.lower()

    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)

    elif optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    elif optimizer_name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr)

    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
