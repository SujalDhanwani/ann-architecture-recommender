# ANN Architecture Recommendation Engine

## Overview

This project implements a modular Artificial Neural Network experimentation framework built from scratch using PyTorch.

The system automatically:

- Preprocesses tabular datasets
- Detects target column
- Detects problem type (Regression / Binary / Multi-class)
- Runs structured ANN architecture experiments
- Applies early stopping
- Compares configurations
- Recommends best architecture
- Retrains and saves the best model

## Features

- Configurable hidden layers
- Configurable activation functions
- Configurable dropout
- Configurable weight initialization
- Controlled experiment pipeline
- Reproducibility (fixed random seeds)
- Validation-based model selection

## How to Run

1. Install dependencies:

    pip install -r requirements.txt


2. Run:

    python main.py  


## Project Structure

models/ -> ANN architecture
training/ -> training loop and evaluation
utils/ -> preprocessing pipeline
main.py -> experiment runner


## What This Project Demonstrates

- Deep understanding of ANN concepts
- Controlled ML experimentation
- Early stopping implementation
- Proper validation strategy
- Architecture comparison logic
- Clean modular design

