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
from models.base_ann import BaseANN
from training.training_loop import train_model
from training.evaluation import evaluate_model

from sklearn.model_selection import train_test_split
import torch
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt


# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


if __name__ == "__main__":

    dataset_path = "data/sample.csv"

    # -------------------------------------------------
    # Dataset Overview
    # -------------------------------------------------
    print("\n==============================")
    print("DATASET OVERVIEW")
    print("==============================")

    df = pd.read_csv(dataset_path)

    print("\nShape:", df.shape)
    print("\nColumns:", list(df.columns))
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nFirst 5 Rows:")
    print(df.head())

    # -------------------------------------------------
    # Preprocessing
    # -------------------------------------------------
    X, y, info = preprocess_dataset(dataset_path)
    problem_type = info["problem_type"]

    print("\n==============================")
    print("DATASET ANALYSIS")
    print("==============================")
    print("Detected Target Column:", info["target"])
    print("Detected Problem Type:", problem_type)
    print("Number of Selected Features:", len(info["features"]))

    # -------------------------------------------------
    # Train / Validation Split
    # -------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Determine output size
    if problem_type == "regression":
        output_size = 1
    elif problem_type == "binary_classification":
        output_size = 1
    else:
        output_size = len(np.unique(y))

    # -------------------------------------------------
    # Grid Search Configuration
    # -------------------------------------------------
    hidden_layer_options = [[32], [64, 32], [128, 64]]
    activation_options = ["relu", "tanh"]
    dropout_options = [0.0, 0.3]
    init_options = ["xavier", "he"]

    results = []

    print("\n==============================")
    print("STARTING ANN ARCHITECTURE SEARCH")
    print("==============================")

    # -------------------------------------------------
    # Grid Search Loop
    # -------------------------------------------------
    for hidden_layers in hidden_layer_options:
        for activation in activation_options:
            for dropout in dropout_options:
                for init_type in init_options:

                    print("\n----------------------------------")
                    print("Testing Configuration:")
                    print("Hidden Layers:", hidden_layers)
                    print("Activation:", activation)
                    print("Dropout:", dropout)
                    print("Initialization:", init_type)

                    model = BaseANN(
                        input_size=X.shape[1],
                        output_size=output_size,
                        hidden_layers=hidden_layers,
                        activation=activation,
                        dropout=dropout,
                        init_type=init_type
                    )

                    model, train_losses, val_losses = train_model(
                        model,
                        X_train,
                        y_train,
                        X_val,
                        y_val,
                        problem_type,
                        epochs=50,
                        patience=5
                    )

                    metrics = evaluate_model(model, X_val, y_val, problem_type)

                    if problem_type == "regression":
                        score = metrics["RMSE"]
                    else:
                        score = metrics["Accuracy"]

                    results.append({
                        "hidden_layers": hidden_layers,
                        "activation": activation,
                        "dropout": dropout,
                        "init": init_type,
                        "score": score
                    })

                    print("Validation Score:", score)

    # -------------------------------------------------
    # Select Best Configuration
    # -------------------------------------------------
    if problem_type == "regression":
        best_config = min(results, key=lambda x: x["score"])
    else:
        best_config = max(results, key=lambda x: x["score"])

    print("\n==============================")
    print("BEST CONFIGURATION FOUND")
    print("==============================")
    print(best_config)

    # -------------------------------------------------
    # Results Summary
    # -------------------------------------------------
    results_df = pd.DataFrame(results)

    print("\nAll Experiment Results (Sorted):")

    if problem_type == "regression":
        print(results_df.sort_values("score"))
    else:
        print(results_df.sort_values("score", ascending=False))

    # Plot architecture comparison
    plt.figure(figsize=(10, 5))
    plt.plot(results_df["score"].values, marker='o')
    plt.title("Architecture Comparison Across Experiments")
    plt.xlabel("Experiment Index")
    plt.ylabel("Score")
    plt.show()

    # -------------------------------------------------
    # Recommendation
    # -------------------------------------------------
    print("\nRecommendation:")
    print(f"For this dataset ({problem_type}), the best ANN architecture is:")
    print(f"Hidden Layers: {best_config['hidden_layers']}")
    print(f"Activation: {best_config['activation']}")
    print(f"Dropout: {best_config['dropout']}")
    print(f"Initialization: {best_config['init']}")
    print("This configuration achieved the best validation performance.")

    # -------------------------------------------------
    # Retrain Best Model
    # -------------------------------------------------
    print("\nRetraining Best Model on Full Dataset...")

    best_model = BaseANN(
        input_size=X.shape[1],
        output_size=output_size,
        hidden_layers=best_config["hidden_layers"],
        activation=best_config["activation"],
        dropout=best_config["dropout"],
        init_type=best_config["init"]
    )

    X_full_train, X_full_val, y_full_train, y_full_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    best_model, train_losses, val_losses = train_model(
        best_model,
        X_full_train,
        y_full_train,
        X_full_val,
        y_full_val,
        problem_type,
        epochs=100,
        patience=10
    )

    # Save model
    torch.save(best_model.state_dict(), "best_model.pth")
    print("Final model saved as best_model.pth")

    # Final evaluation
    final_metrics = evaluate_model(best_model, X_full_val, y_full_val, problem_type)

    print("\nFinal Model Metrics:")
    for k, v in final_metrics.items():
        print(f"{k}: {v}")

    # Plot convergence
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Best Model Convergence")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
