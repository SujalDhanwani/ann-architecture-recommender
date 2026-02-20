import torch
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

from utils.data_loader import preprocess_dataset
from experiment_engine import run_experiment

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
    # RAW Dataset Summary
    # ------------------------------------------------------
    print("\n=================================================")
    print("RAW DATASET SUMMARY")
    print("=================================================")

    try:
        df_raw = pd.read_csv(dataset_path)
    except Exception:
        df_raw = pd.read_csv(dataset_path, encoding="latin1", engine="python")

    print("\nShape:", df_raw.shape)
    print("\nColumns:", list(df_raw.columns))
    print("\nMissing Values:\n", df_raw.isnull().sum())
    print("\nHead:")
    print(df_raw.head())

    # ------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------
    print("\n=================================================")
    print("PREPROCESSING")
    print("=================================================")

    X, y, info = preprocess_dataset(dataset_path)

    print("\nDetected Target Column :", info["target"])
    print("Detected Problem Type  :", info["problem_type"])
    print("Number of Features     :", len(info["features"]))
    print("\nX Shape:", X.shape)
    print("y Shape:", y.shape)

    problem_type = info["problem_type"]

    # ------------------------------------------------------
    # Strategy Selection
    # ------------------------------------------------------
    print("\nChoose Search Strategy:")
    print("1 → Grid Search")
    print("2 → Optuna Bayesian Optimization")

    choice = input("Enter choice (1 or 2): ").strip()
    search_strategy = "optuna" if choice == "2" else "grid"

    print(f"\nRunning {search_strategy.upper()}...\n")

    # Run Experiment
    results = run_experiment(
        dataset_path,
        search_strategy=search_strategy
    )

    # ------------------------------------------------------
    # GRID SEARCH RESULTS
    # ------------------------------------------------------
    if results["strategy"] == "grid":

        results_df = results["results_df"]
        best_config = results["best_config"]

        print("\n===== BEST CONFIGURATION =====")
        print(best_config)

        # sort
        if problem_type == "regression":
            results_df = results_df.sort_values("score")
        else:
            results_df = results_df.sort_values("score", ascending=False)

        print("\n===== ALL RESULTS =====")
        print(results_df)

        # ---------- PLOT 1: Score Trend ----------
        plt.figure(figsize=(10, 5))
        plt.plot(results_df["score"].values, marker="o")
        plt.title("Score vs Experiment Index")
        plt.xlabel("Experiment")
        plt.ylabel("Score")
        plt.grid(True)
        plt.show()

        # ---------- PLOT 2: Dropout vs Score ----------
        plt.figure(figsize=(7, 5))
        plt.scatter(results_df["dropout"], results_df["score"])
        plt.title("Dropout vs Score")
        plt.xlabel("Dropout")
        plt.ylabel("Score")
        plt.grid(True)
        plt.show()

    # ------------------------------------------------------
    # OPTUNA RESULTS
    # ------------------------------------------------------
    else:
        study = results["study"]
        best_params = results["best_config"]
        best_score = results["best_score"]

        print("\n===== BEST PARAMETERS =====")
        for k, v in best_params.items():
            print(f"{k}: {v}")

        print("\nBest Score:", best_score)

        trial_scores = [t.value for t in study.trials]

        # ---------- PLOT 1: Trial Score ----------
        plt.figure(figsize=(10, 5))
        plt.plot(trial_scores, marker="o")
        plt.title("Optuna Trial Scores")
        plt.xlabel("Trial")
        plt.ylabel("Objective Score")
        plt.grid(True)
        plt.show()

        # ---------- PLOT 2: Best Score Progression ----------
        best_progression = []
        best_so_far = float("inf")

        for s in trial_scores:
            if s < best_so_far:
                best_so_far = s
            best_progression.append(best_so_far)

        plt.figure(figsize=(10, 5))
        plt.plot(best_progression)
        plt.title("Best Score Progression")
        plt.xlabel("Trial")
        plt.ylabel("Best Score So Far")
        plt.grid(True)
        plt.show()

    print("\n=================================================")
    print("EXPERIMENT COMPLETED")
    print("=================================================")