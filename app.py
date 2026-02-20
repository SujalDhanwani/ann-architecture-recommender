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

st.title("🧠 ANN Architecture Recommendation Engine")
st.write("Upload your dataset and automatically discover the best ANN configuration.")

# Initialize log storage
if "log" not in st.session_state:
    st.session_state.log = []


# -------------------------------------------------
# File Upload
# -------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file:

    # Safe CSV loading
    try:
        df = pd.read_csv(uploaded_file)
    except:
        df = pd.read_csv(uploaded_file, encoding="latin1", engine="python")

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    temp_path = "data/temp_dataset.csv"
    df.to_csv(temp_path, index=False)

    # Preprocess
    try:
        X, y, info = preprocess_dataset(temp_path)
    except Exception as e:
        st.error(f"❌ Preprocessing Failed: {e}")
        st.stop()

    st.subheader("📌 Dataset Analysis")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Features Used", len(info["features"]))

    st.write("**Target Column →**", info["target"])
    st.write("**Problem Type →**", info["problem_type"])


    # -------------------------------------------------
    # Search Strategy
    # -------------------------------------------------
    st.subheader("⚙ Choose Search Strategy")

    strategy = st.selectbox(
        "Search Method",
        ["Grid Search", "Optuna Bayesian Optimization"]
    )

    n_trials = None
    if strategy == "Optuna Bayesian Optimization":
        n_trials = st.slider("Optuna Trials", 5, 40, 15)


    # -------------------------------------------------
    # Run Experiment
    # -------------------------------------------------
    if st.button("🚀 Run ANN Experiment"):

        st.subheader("📡 Live Logs")
        log_box = st.empty()
        progress_bar = st.progress(0)
        st.session_state.log = []

        def stream_log(msg: str):
            """Display last 15 log lines safely."""
            st.session_state.log.append(msg)
            log_box.markdown("```\n" + "\n".join(st.session_state.log[-15:]) + "\n```")

        stream_log("Experiment Started...")

        start = time.time()

        # ---------------------------
        # Run GRID
        # ---------------------------
        if strategy == "Grid Search":

            try:
                result = run_experiment(
                    temp_path,
                    search_strategy="grid",
                    callback_fn=stream_log
                )
            except Exception as e:
                st.error(f"❌ Grid Search Failed: {e}")
                st.stop()

            progress_bar.progress(100)
            stream_log("Grid Search Completed.")

        # ---------------------------
        # Run OPTUNA
        # ---------------------------
        else:
            current_trial = {"count": 0}

            def wrapped_callback(msg):
                stream_log(msg)
                current_trial["count"] += 1
                step = min(current_trial["count"] / n_trials, 1.0)
                progress_bar.progress(step)

            try:
                result = run_experiment(
                    temp_path,
                    search_strategy="optuna",
                    callback_fn=wrapped_callback,
                    n_trials=n_trials
                )
            except Exception as e:
                st.error(f"❌ Optuna Failed: {e}")
                st.stop()

            stream_log("Optuna Optimization Completed.")

        # ---------------------------
        # Total Time
        # ---------------------------
        total_time = time.time() - start
        mins, secs = int(total_time // 60), int(total_time % 60)

        st.success(f"🎉 Completed in {mins} min {secs} sec")

        # ---------------------------
        # BEST RESULT OUTPUT
        # ---------------------------
        st.subheader("🏆 Best Configuration")
        st.json(result["best_config"])

        # ---------------------------------
        # Score Label (Adaptive)
        # ---------------------------------
        if info["problem_type"] == "regression":
            score_label = "Best RMSE"
        else:
            score_label = "Best Accuracy"

        if result["strategy"] == "optuna":
            st.metric(score_label, value=round(result["best_score"], 4))

        if result["strategy"] == "grid":
            st.write("Grid search completed. Best configuration displayed above.")