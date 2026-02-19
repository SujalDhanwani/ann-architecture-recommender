# from utils.data_loader import load_csv, detect_column_types, detect_target_column, detect_problem_type,select_features,handle_missing_values,encode_categorical_columns,extract_datetime_features,scale_numeric_columns

# if __name__ == "__main__":
#     path = "data/sample.csv"   # Change path as needed

#     df = load_csv(path)
#     print("\n Data loaded successfully\n")
#     print(df.head())

#     # Column type detection
#     print("\n Detecting column types...")
#     column_types = detect_column_types(df)
#     print("Detected column types:", column_types)

#     # Target detection
#     print("\n Detecting target column...")
#     target_column = detect_target_column(df)
#     print("Detected target column:", target_column)

#     # Problem type detection
#     print("\n Detecting problem type...")
#     problem_type = detect_problem_type(df, target_column)
#     print("Detected problem type:", problem_type)

#     print("\n Selecting feature columns...")
#     feature_columns = select_features(df, target_column)
#     print("Selected feature columns:", feature_columns)
    
#     df = handle_missing_values(df, column_types)
#     print("Missing values handled")
#     print(df.isna().sum())
    
#     df = encode_categorical_columns(df, column_types)
#     print("encoding done")
    
#     print(df.head())

    
#     df = extract_datetime_features(df, column_types)
#     print("Datetime features extracted")
#     print(df.head())
    
#     df = scale_numeric_columns(df, column_types)
#     print("Numeric features scaled")
#     print(df.head())

from utils.data_loader import preprocess_dataset
from experiment_engine import run_experiment

import torch
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================================
# Reproducibility
# ==========================================================
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":

    dataset_path = "data/sample.csv"

    # ------------------------------------------------------
    # Dataset Overview
    # ------------------------------------------------------
    print("\n=================================================")
    print("DATASET OVERVIEW")
    print("=================================================")

    df = pd.read_csv(dataset_path)

    print("\nShape:", df.shape)
    print("\nColumns:", list(df.columns))
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nFirst 5 Rows:")
    print(df.head())

    # ------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------
    X, y, info = preprocess_dataset(dataset_path)
    problem_type = info["problem_type"]

    print("\n=================================================")
    print("DATASET ANALYSIS")
    print("=================================================")
    print("Detected Target Column :", info["target"])
    print("Detected Problem Type  :", problem_type)
    print("Number of Features     :", len(info["features"]))

    # ------------------------------------------------------
    # Choose Search Strategy
    # ------------------------------------------------------
    print("\n1 → Grid Search")
    print("2 → Optuna")

    choice = input("Enter choice (1 or 2): ").strip()
    search_strategy = "optuna" if choice == "2" else "grid"

    print(f"\nRunning {search_strategy.upper()} search...\n")

    results = run_experiment(dataset_path, search_strategy=search_strategy)

    # ======================================================
    # GRID SEARCH VISUALIZATION
    # ======================================================
    if results["strategy"] == "grid":

        results_df = results["results_df"]
        best_config = results["best_config"]

        print("\nBest Configuration:")
        print(best_config)

        # Sort properly
        if problem_type == "regression":
            results_df = results_df.sort_values("score")
        else:
            results_df = results_df.sort_values("score", ascending=False)

        print("\nAll Results:")
        print(results_df)

        # -----------------------------
        # Plot 1: Score vs Experiment
        # -----------------------------
        plt.figure(figsize=(10, 5))
        plt.plot(results_df["score"].values, marker='o')
        plt.title("Experiment Score Trend")
        plt.xlabel("Experiment Index")
        plt.ylabel("Score")
        plt.grid(True)
        plt.show()

        # -----------------------------
        # Plot 2: Hidden Layer Size vs Score
        # -----------------------------
        hidden_sizes = results_df["hidden_layers"].astype(str)
        plt.figure(figsize=(10, 5))
        plt.scatter(hidden_sizes, results_df["score"])
        plt.title("Hidden Layer Configuration vs Score")
        plt.xlabel("Hidden Layers")
        plt.ylabel("Score")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()

        # -----------------------------
        # Plot 3: Dropout vs Score
        # -----------------------------
        plt.figure(figsize=(8, 5))
        plt.scatter(results_df["dropout"], results_df["score"])
        plt.title("Dropout vs Score")
        plt.xlabel("Dropout Rate")
        plt.ylabel("Score")
        plt.grid(True)
        plt.show()

        # -----------------------------
        # Plot 4: Activation vs Score
        # -----------------------------
        activations = results_df["activation"].unique()
        for act in activations:
            subset = results_df[results_df["activation"] == act]
            plt.scatter(
                [act] * len(subset),
                subset["score"]
            )

        plt.title("Activation Function Impact")
        plt.xlabel("Activation")
        plt.ylabel("Score")
        plt.grid(True)
        plt.show()

    # ======================================================
    # OPTUNA VISUALIZATION
    # ======================================================
    elif results["strategy"] == "optuna":

        study = results["study"]
        best_params = results["best_config"]
        best_score = results["best_score"]

        print("\nBest Parameters:")
        for k, v in best_params.items():
            print(f"{k}: {v}")

        print("\nBest Score:", best_score)

        # -----------------------------
        # Plot 1: Score vs Trial
        # -----------------------------
        trial_scores = [trial.value for trial in study.trials]

        plt.figure(figsize=(10, 5))
        plt.plot(trial_scores, marker='o')
        plt.title("Optuna Trial Scores")
        plt.xlabel("Trial")
        plt.ylabel("Objective Value")
        plt.grid(True)
        plt.show()

        # -----------------------------
        # Plot 2: Best Score Progression
        # -----------------------------
        best_progression = []
        best_so_far = float("inf")

        for val in trial_scores:
            if val < best_so_far:
                best_so_far = val
            best_progression.append(best_so_far)

        plt.figure(figsize=(10, 5))
        plt.plot(best_progression)
        plt.title("Best Score Progression")
        plt.xlabel("Trial")
        plt.ylabel("Best Objective So Far")
        plt.grid(True)
        plt.show()

    print("\n=================================================")
    print("EXPERIMENT COMPLETED")
    print("=================================================")
