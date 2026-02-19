import streamlit as st
import pandas as pd
import time

from utils.data_loader import preprocess_dataset
from experiment_engine import run_experiment

# -------------------------------------------------
# Streamlit Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="ANN Architecture Recommender",
    layout="wide"
)

st.title("ANN Architecture Recommendation Engine")
st.write("Upload your dataset and the system will automatically search for the best ANN architecture.")

# Initialize log storage
if "log" not in st.session_state:
    st.session_state.log = []

# -------------------------------------------------
# File Upload
# -------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding_errors='replace')
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    temp_path = "data/temp_dataset.csv"
    df.to_csv(temp_path, index=False)

    # Preprocess
    X, y, info = preprocess_dataset(temp_path)

    st.subheader("📌 Dataset Analysis")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Features Used", len(info["features"]))

    st.write("**Target Column:**", info["target"])
    st.write("**Problem Type:**", info["problem_type"])

    # -------------------------------------------------
    # Choose Search Strategy
    # -------------------------------------------------
    st.subheader("⚙ Choose Search Strategy")

    strategy = st.selectbox(
        "Search Method",
        ["Grid Search", "Optuna Bayesian Optimization"]
    )

    n_trials = None
    if strategy == "Optuna Bayesian Optimization":
        n_trials = st.slider("Number of Optuna Trials", 5, 40, 15)

    # -------------------------------------------------
    # Run Button
    # -------------------------------------------------
    if st.button(" Run ANN Experiment"):

        st.subheader(" Live Experiment Logs")

        # Log box + Progress bar
        log_box = st.empty()
        progress_bar = st.progress(0)
        st.session_state.log = []

        start_time = time.time()

        # -------------------------------
        # Logging function for UI
        # -------------------------------
        def stream_log(msg):
            st.session_state.log.append(msg)
            log_box.markdown("```\n" + "\n".join(st.session_state.log[-15:]) + "\n```")

        stream_log("Experiment Started...")

        # -------------------------------
        # Run Grid Search
        # -------------------------------
        if strategy == "Grid Search":
            result = run_experiment(
                temp_path,
                search_strategy="grid",
                callback_fn=stream_log
            )
            progress_bar.progress(100)
            stream_log("Grid Search Completed.")

        # -------------------------------
        # Run Optuna
        # -------------------------------
        else:
            current_trial = {"num": 0}

            def wrapped_callback(text):
                """Receive messages from Optuna and show live."""
                stream_log(text)
                current_trial["num"] += 1
                progress = min(current_trial["num"] / n_trials, 1.0)
                progress_bar.progress(progress)

            result = run_experiment(
                temp_path,
                search_strategy="optuna",
                callback_fn=wrapped_callback,
                n_trials=n_trials
            )

            stream_log("Optuna Optimization Completed.")

        # -------------------------------
        # Total Time
        # -------------------------------
        total_time = time.time() - start_time
        mins = int(total_time // 60)
        secs = int(total_time % 60)

        st.success(f"🎉 Completed in {mins} minutes {secs} seconds!")

        # -------------------------------
        # Final Output
        # -------------------------------
        st.subheader(" Best Configuration")

        if result["strategy"] == "grid":
            st.json(result["best_config"])
            st.write("Grid Search complete.")

        else:
            st.json(result["best_config"])
            st.write("Best Score:", result["best_score"])

