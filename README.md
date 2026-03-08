# Charukhesh B R -- AE22B028

### [W&B Report Link](https://wandb.ai/charukhesh-indian-institute-of-technology-madras/da6401_assignment_1-src/reports/W-B-Report-AE22B028--VmlldzoxNjEzMzIzNA)

### [Github Repo Link](https://github.com/Charukhesh/da6401_assignment_1.git)

# DA6401 Assignment -- Neural Network from Scratch

This repository contains an implementation of a fully connected neural network (Multi-Layer Perceptron) built from scratch using NumPy, along with experiments and analyses performed as part of the DA6401 Deep Learning assignment.

The project explores neural network fundamentals including forward and backward propagation, optimization algorithms, activation functions, and hyperparameter tuning. Various experiments were conducted to study training behavior, optimization dynamics, and model generalization.

# Repository Structure

```
DA6401_Assignment/
│
├── model/                         # Saved model artifacts (optional)
│
├── src/
│   │
│   ├── ann/                       # Core neural network implementation
│   │   ├── __init__.py
│   │   ├── activations.py         # ReLU, Sigmoid, Tanh implementations
│   │   ├── neural_layer.py        # Dense layer implementation
│   │   ├── neural_network.py      # Main NeuralNetwork class
│   │   ├── objective_functions.py # MSE and Cross Entropy loss
│   │   └── optimizers.py          # SGD, Momentum, NAG, RMSProp
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   └── data_loader.py         # Dataset loading utilities
│   │
│   ├── activation_comparison.py   # ReLU vs Tanh experiment
│   ├── loss_comparison.py         # MSE vs Cross Entropy experiment
│   ├── optimizer_comparison.py    # Optimizer performance comparison
│   ├── symmetry_grad.py           # Weight initialization symmetry experiment
│   ├── vanishing_grad.py          # Vanishing gradient analysis
│   ├── error_analysis.py          # Confusion matrix and failure visualization
│   ├── fashion_transfer.py        # Fashion-MNIST transfer experiment
│   │
│   ├── train.py                   # Model training script
│   └── inference.py               # Model evaluation script
│
├── best_model.npy                 # Saved trained model weights
├── best_config.json               # Configuration of best model
│
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

# Installation

Clone the repository:
```bash
git clone https://github.com/Charukhesh/da6401_assignment_1.git 
cd da6401_assignment_1
```

Install dependencies:
```bash
pip install -r requirements.txt
```

# Training the Model

To train the neural network:
```bash
python src/train.py
```

Example with custom hyperparameters:
```bash
python src/train.py \
--dataset mnist \
--epochs 10 \
--batch_size 64 \
--optimizer rmsprop \
--learning_rate 0.001 \
--activation relu \
--hidden_size 128 86 64
```

Training will output:

- Epoch losses  
- Validation accuracy  
- Saved model weights  

Saved artifacts:
```
best_model.npy  
best_config.json
```

# Running Inference

To evaluate the saved model:
```bash
python src/inference.py
```

Outputs:

- Accuracy  
- Precision  
- Recall  
- F1 Score  

# Experiments

The repository includes several experiment scripts used to analyze network behavior.

## Activation Function Comparison

Compare convergence behavior of ReLU and Tanh.
```bash
python src/activation_comparison.py
```

## Loss Function Comparison

Compare MSE vs Cross Entropy.
```bash
python src/loss_comparison.py
```

## Optimizer Comparison

Evaluate convergence rates of:

- SGD  
- Momentum  
- NAG  
- RMSProp  

```bash
python src/optimizer_comparison.py
```

## Symmetry Breaking Experiment

Demonstrates why zero initialization prevents learning.
```bash
python src/symmetry_grad.py
```

## Vanishing Gradient Analysis

Compares gradient norms between:

- Sigmoid  
- ReLU  
```bash
python src/vanishing_grad.py
```

## Error Analysis

Produces:

- Confusion matrix  
- Misclassified sample visualization  
```bash
python src/error_analysis.py
```

## Fashion-MNIST Transfer Study

Evaluates transfer of MNIST hyperparameters to Fashion-MNIST.
```bash
python src/fashion_transfer.py
```

# Dataset

The project supports two datasets.

## MNIST

Handwritten digit dataset used for training experiments.

## Fashion-MNIST

Clothing classification dataset used for transfer learning experiments.

Both datasets are automatically downloaded through the data loader.



