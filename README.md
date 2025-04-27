# Hyperparameter Optimization for CNN on CIFAR-10

This project explores hyperparameter optimization for a convolutional neural network (CNN) on the CIFAR-10 dataset using PyTorch and Optuna. The model is trained on a reduced version of the dataset (1K train, 300 validation) to test optimization efficiency.

## 🚀 Results
- Achieved 78%+ validation accuracy in under 10 epochs
- Reduced training time by over 40% using Optuna’s pruning mechanism

## 🔧 Technologies
- PyTorch
- Optuna
- torchvision

## 📁 Structure
- `cnn_model.py`: CNN architecture definition
- `optuna_tuner.py`: Hyperparameter search and training loop
- `run_experiment.ipynb`: Jupyter Notebook version of the training pipeline

## 📊 Features
- Dropout rate tuning
- Optimizer selection (Adam, SGD, RMSprop)
- Learning rate range search
- Pruned trials for faster convergence

## 🧠 Future Work
- Expand to full dataset training
- Try other NAS libraries like Ray Tune or Hyperopt
