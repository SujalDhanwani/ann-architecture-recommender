import torch
import torch.nn as nn

def get_loss_function(problem_type):
    if problem_type =="classification":
        return nn.CrossEntropyLoss()
    elif problem_type=="regression":
        return nn.MSELoss()
    
    elif problem_type=="multilabel_classification":
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")

def get_optimizer(model, lr = 0.001):
    return torch.optim.Adam(model.parameters(), lr=lr)

