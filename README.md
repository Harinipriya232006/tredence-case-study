# Self-Pruning Neural Network (CIFAR-10)

This project implements a self-pruning neural network using PyTorch. The model learns to automatically remove unnecessary weights during training using learnable gates and L1 regularization.

---

## Project Overview

The goal of this project is to build a neural network that can:

- Learn important weights
- Remove unimportant connections
- Maintain good accuracy while becoming sparse

The model is trained on the CIFAR-10 dataset.

---

## Project Structure

self_pruning_network/
├── model.py
├── train.py
├── report.md
├── gate_distribution.png

---

## Installation

1. Clone the repository:

git clone https://github.com/YOUR_USERNAME/tredence-case-study.git

2. Navigate into the project folder:

cd tredence-case-study

3. Install required libraries:

pip install torch torchvision matplotlib

---

## How to Run

Run the training script:

python train.py

---

## Model Description

The model uses a custom layer called `PrunableLinear`.

Each weight is controlled by a gate:

Effective Weight = Weight × Sigmoid(Gate)

- Sigmoid keeps values between 0 and 1
- L1 regularization pushes gates toward 0
- This removes unnecessary weights

---

## Experiments

The model is trained using different lambda (λ) values:

| Lambda | Accuracy (%) | Sparsity (%) |
|--------|-------------|--------------|
| 0.0001 |             |              |
| 0.001  |             |              |
| 0.01   |             |              |

(Fill these after running the code)

---

## Results

- Higher λ → More pruning (high sparsity)
- Lower λ → Better accuracy
- Best model balances both

---

## Output

The project generates:

- Accuracy and sparsity results
- Gate distribution plot (gate_distribution.png)

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- Matplotlib

---

## Future Improvements

- Add GPU support
- Try advanced pruning methods
- Improve accuracy
- Deploy model

---

## Author

Harini Priya
