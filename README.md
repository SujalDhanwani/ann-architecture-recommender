#  ANN Architecture Recommendation Engine

A complete **machine learning experimentation system** built from scratch using **PyTorch**, designed to automatically search, evaluate, and recommend the best Artificial Neural Network (ANN) architecture for any tabular dataset.

This project mirrors **real ML engineering workflows**:
**preprocessing → experiment engine → optimization → evaluation → best-model selection → UI**

---

##  Key Features

###  Automatic Dataset Understanding
- Detects **target column**
- Determines **problem type**
  - Regression  
  - Binary Classification  
  - Multi-class Classification
- Intelligent feature selection
- Handles missing values
- Safe categorical encoding
- Safe numerical scaling
- Date-Time feature extraction

---

###  ANN Experimentation Engine
- Configurable hidden layers
- Activation functions (ReLU, Tanh)
- Dropout regularization
- Weight initialization (Xavier / He)
- Multiple optimizers (Adam / SGD / RMSProp)
- Early stopping
- Reproducibility (fixed seeds)

---

##  Experiment Strategies
- **Grid Search**
- **Optuna Bayesian Optimization**
- Callbacks for real-time logging

---

##  Streamlit UI (Optional)
- Upload dataset  
- Live experiment logs  
- Progress bar  
- Results summary  
- Best architecture display  

---

##  Project Structure

```plaintext
models/
    base_ann.py            → ANN model architecture

training/
    training_loop.py       → training engine + early stopping
    evaluation.py          → metrics (RMSE, MAE, Accuracy, F1)

utils/
    data_loader.py         → preprocessing pipeline (encoding, scaling, datetime)

experiment_engine.py       → grid search + optuna optimization
main.py                    → CLI experiment runner
app.py                     → Streamlit UI (interactive mode)

requirements.txt           → dependencies
README.md                  → project documentation
```

---

##  Installation

```bash
pip install -r requirements.txt
```

---

##  Running the CLI

```bash
python main.py
```

You will be prompted to choose:

1️ **Grid Search**  
2️ **Optuna Bayesian Optimization**

The system will train multiple architectures and select the best one.

---

## Running the UI (Streamlit)

```bash
streamlit run app.py
```

### Features in UI:
- Dataset preview  
- Automatic analysis  
- Live logging  
- Real-time trial updates  
- Best architecture summary  

---

##  What This Project Demonstrates

- Deep understanding of ANN architecture design  
- Ability to build full ML pipelines  
- Preprocessing automation for real datasets  
- Engineering-focused experiment management  
- Hyperparameter optimization (Grid + Bayesian)  
- Strong ML code organization principles  
- Real-world early stopping strategy  
- End-to-end model selection workflow  

This project is designed as a **learning-focused ML engineering system**, not just a model script.  
It reflects how real teams build **experiment engines inside AI companies**.
