# 🧠 ANN Architecture Recommendation Engine

## Overview

This project implements a **modular Artificial Neural Network (ANN) experimentation framework** built entirely from scratch using **PyTorch**.

The system automatically:

- Loads and preprocesses tabular datasets  
- Detects the target column  
- Detects the problem type (Regression / Binary / Multi-class)  
- Runs **Grid Search** or **Optuna Bayesian Optimization**  
- Applies **early stopping**  
- Compares architectures using validation performance  
- Recommends the **best ANN architecture**  
- Retrains and saves the final best model  

---

## 🚀 Features

- Configurable hidden layers  
- Configurable activation functions  
- Dropout support  
- Xavier & He initialization  
- Multiple optimizers (Adam / SGD / RMSProp)  
- Early stopping implementation  
- Scaled regression targets  
- Reproducible experiments (fixed random seeds)  
- Fully automated preprocessing pipeline  
- Modular ML engineering system  

---

## 🔧 How to Run

### 1. Install dependencies
pip install -r requirements.txt

2. Run experiment system (CLI)
    python main.py

You will be prompted to select:

Grid Search

Optuna Bayesian Optimization

📁 Project Structure


models/
    base_ann.py        → ANN model architecture

training/
    training_loop.py   → training + early stopping
    evaluation.py      → metrics (RMSE/MAE/Accuracy/F1)

utils/
    data_loader.py     → preprocessing pipeline (encoding, scaling, datetime)

experiment_engine.py   → grid search + optuna experiments
main.py                → CLI experiment runner
app.py                 → Streamlit UI (optional)


## What This Project Demonstrates

. Deep understanding of ANN architecture

. Ability to build a full ML experimentation system

. Proper validation & early stopping

. Grid search & Bayesian optimization

. Engineering-focused modular design

. End-to-end preprocessing automation


