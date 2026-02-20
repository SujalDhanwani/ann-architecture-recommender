import optuna
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split

from models.base_ann import BaseANN
from training.training_loop import train_model
from training.evaluation import evaluate_model
from utils.data_loader import preprocess_dataset
import torch


# =====================================================================
# GRID SEARCH
# =====================================================================
def run_grid_search(file_path, callback_fn=None):

    start_total = time.time()

    X, y, info = preprocess_dataset(file_path)
    problem_type = info["problem_type"]
    y_scaler = info.get("y_scaler")

    # ---- Train / Validation Split ----
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Output size
    if problem_type in ["regression", "binary_classification"]:
        output_size = 1
    else:
        output_size = len(np.unique(y_train))

    # Search space
    hidden_layer_options = [[32], [64, 32], [128, 64]]
    activation_options = ["relu", "tanh"]
    dropout_options = [0.0, 0.3]
    init_options = ["xavier", "he"]

    total_experiments = (
        len(hidden_layer_options)
        * len(activation_options)
        * len(dropout_options)
        * len(init_options)
    )

    results = []
    exp_counter = 1

    for hidden_layers in hidden_layer_options:
        for activation in activation_options:
            for dropout in dropout_options:
                for init_type in init_options:

                    if callback_fn:
                        callback_fn(
                            f"[Exp {exp_counter}/{total_experiments}] "
                            f"HL={hidden_layers}, Act={activation}, Drop={dropout}, Init={init_type}"
                        )

                    start_exp = time.time()

                    model = BaseANN(
                        input_size=X.shape[1],
                        output_size=output_size,
                        hidden_layers=hidden_layers,
                        activation=activation,
                        dropout=dropout,
                        init_type=init_type
                    )

                    model = train_model(
                        model,
                        X_train,
                        y_train,
                        X_val,
                        y_val,
                        problem_type,
                        epochs=60,
                        patience=8,
                    )

                    metrics = evaluate_model(
                        model, X_val, y_val, problem_type, y_scaler=y_scaler
                    )

                    score = (
                        metrics["RMSE"]
                        if problem_type == "regression"
                        else metrics["Accuracy"]
                    )

                    elapsed = time.time() - start_exp

                    if callback_fn:
                        callback_fn(f"Score: {score:.4f} | Time: {elapsed:.2f}s")

                    results.append({
                        "hidden_layers": hidden_layers,
                        "activation": activation,
                        "dropout": dropout,
                        "init": init_type,
                        "score": score
                    })

                    exp_counter += 1

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    total_time = time.time() - start_total

    if callback_fn:
        callback_fn(f"Grid Search Completed in {total_time/60:.2f} minutes.")

    results_df = pd.DataFrame(results)

    # Choose best config
    best_config = (
        min(results, key=lambda x: x["score"])  # RMSE → lower is better
        if problem_type == "regression"
        else max(results, key=lambda x: x["score"])
    )

    return {
        "strategy": "grid",
        "results_df": results_df,
        "best_config": best_config,
        "problem_type": problem_type
    }


# =====================================================================
# OPTUNA SEARCH
# =====================================================================
def run_optuna_search(file_path, n_trials=15, callback_fn=None):

    start_total = time.time()

    X, y, info = preprocess_dataset(file_path)
    problem_type = info["problem_type"]
    y_scaler = info.get("y_scaler")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if problem_type in ["regression", "binary_classification"]:
        output_size = 1
    else:
        output_size = len(np.unique(y_train))

    # Optuna objective
    def objective(trial):

        hidden1 = trial.suggest_int("hidden1", 16, 64)
        hidden2 = trial.suggest_int("hidden2", 8, 32)
        dropout = trial.suggest_float("dropout", 0.0, 0.4)
        activation = trial.suggest_categorical("activation", ["relu", "tanh"])
        lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
        optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"])

        start_t = time.time()

        model = BaseANN(
            input_size=X.shape[1],
            output_size=output_size,
            hidden_layers=[hidden1, hidden2],
            activation=activation,
            dropout=dropout,
            init_type="he"
        )

        model = train_model(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            problem_type,
            epochs=60,
            patience=8,
            lr=lr,
            optimizer_name=optimizer_name
        )

        metrics = evaluate_model(
            model, X_val, y_val, problem_type, y_scaler=y_scaler
        )

        score = (
            metrics["RMSE"]
            if problem_type == "regression"
            else 1 - metrics["Accuracy"]
        )

        elapsed = time.time() - start_t

        if callback_fn:
            callback_fn(
                f"Trial {trial.number+1}/{n_trials} "
                f"| Score={score:.4f} | Time={elapsed:.2f}s"
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return score

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    total_time = time.time() - start_total

    if callback_fn:
        callback_fn(f"Optuna Completed in {total_time/60:.2f} minutes.")

    return {
        "strategy": "optuna",
        "study": study,
        "best_config": study.best_trial.params,
        "best_score": study.best_trial.value,
        "problem_type": problem_type
    }


# =====================================================================
# Unified Entry Point
# =====================================================================
def run_experiment(file_path, search_strategy="grid", callback_fn=None, n_trials=15):

    if search_strategy == "grid":
        return run_grid_search(file_path, callback_fn=callback_fn)

    elif search_strategy == "optuna":
        return run_optuna_search(
            file_path,
            n_trials=n_trials,
            callback_fn=callback_fn
        )

    else:
        raise ValueError("Unsupported search strategy.")