#  ANN Architecture Recommendation Engine

A complete **machine learning experimentation system** built from scratch using **PyTorch**, designed to automatically search, evaluate, and recommend the best Artificial Neural Network (ANN) architecture for any tabular dataset.

This project mirrors **real ML engineering workflows**:  
**Preprocessing → Experiment Engine → Optimization → Evaluation → Best-Model Selection → UI**

---

##  Key Features

###  Automatic Dataset Understanding
- Detects **target column**
- Identifies **problem type**:
  - Regression  
  - Binary Classification  
  - Multi-Class Classification
- Safe preprocessing:
  - Missing value handling  
  - Categorical encoding  
  - Numerical scaling  
  - Date-Time feature extraction  
- Smart feature selection

---

## 🧠 ANN Experimentation Engine
Designed for flexibility, control, and performance.

- Customizable **hidden layers**
- Activation functions: **ReLU**, **Tanh**
- **Dropout** regularization
- Weight initialization: **Xavier**, **He**
- Optimizers: **Adam**, **SGD**, **RMSProp**
- Early stopping for stable training
- Full reproducibility with fixed seeds

---

## 🔬 Experiment Strategies

### 1️ **Grid Search**
Traditional exhaustive search over defined hyperparameters.

### 2️ **Optuna Bayesian Optimization**
Efficient, intelligent search for:
- Learning rate  
- Layer sizes  
- Activation  
- Dropout  
- Optimizer  
- Batch size  

Real-time logging through callbacks.

---

## 📊 Streamlit UI (Optional)
A clean, beginner-friendly interface.

- Upload your dataset  
- Automatic analysis  
- Experiment selection (Grid / Optuna)  
- Live logs and progress  
- Real-time trial updates  
- Final architecture summary  

Run with:

```bash
streamlit run app.py
🗂 Project Structure
models/
    base_ann.py            → ANN model architecture

training/
    training_loop.py       → training engine + early stopping
    evaluation.py          → metrics (RMSE, MAE, Accuracy, F1)

utils/
    data_loader.py         → preprocessing pipeline

experiment_engine.py       → Grid Search + Optuna optimization
main.py                    → CLI experiment runner
app.py                     → Streamlit UI

requirements.txt           → dependencies
README.md                  → documentation

```

Installation

```bash
pip install -r requirements.txt
```

Running the CLI

```bash
python main.py
```

You will be prompted to choose:

- **Grid Search**
- **Optuna Bayesian Optimization**

The system will train multiple architectures and automatically select the best-performing model based on the evaluation metric.

---

## What This Project Demonstrates

- Solid understanding of ANN architecture design  
- Clean, end-to-end ML pipeline  
- Professional preprocessing automation  
- Handling of real-world datasets  
- Efficient hyperparameter optimization  
- Modular and maintainable code structure  
- Real machine learning experiment workflow  
- Early stopping and safe training patterns  

This project is built as a learning-focused ML engineering system, similar to what real AI teams develop for internal experimentation.

---
