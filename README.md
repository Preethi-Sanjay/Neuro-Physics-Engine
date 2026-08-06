# NeuroPhysicsEngine

A modular Physics-Informed Neural Network (PINN) framework built with PyTorch for solving partial differential equations using automatic differentiation and physics-constrained optimization.

---

## Overview

NeuroPhysicsEngine is a scientific machine learning framework that solves partial differential equations without requiring labeled simulation data. Instead of learning purely from datasets, the network embeds the governing physical equations directly into the loss function.

The project is designed with a modular architecture so new PDEs can be added with minimal changes to the training pipeline.

---

## Features

- Modular PINN architecture
- Automatic differentiation with PyTorch
- Physics, initial, and boundary condition losses
- Dynamic equation loading
- GPU support (CUDA)
- Scientific visualization
- Multiple evaluation metrics

---

## Supported Equations

| Equation | Status |
|----------|--------|
| Heat Equation | ✓ |
| Wave Equation | ✓ |
| Burgers Equation | ✓ |
| Laplace Equation | Planned |
| Poisson Equation | Planned |

---

## Project Structure

```text
NeuroPhysicsEngine/
│
├── checkpoints/
├── configs/
├── equations/
├── examples/
├── losses/
├── models/
├── outputs/
├── solvers/
├── utils/
│
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

---

## PINN Workflow

```text
Input (x,t)
      │
      ▼
Neural Network
      │
      ▼
Automatic Differentiation
      │
      ▼
Physics Residual
      │
      ▼
Physics + Initial + Boundary Loss
      │
      ▼
Backpropagation
      │
      ▼
Prediction
```

---

## Training

Heat Equation

```bash
python train.py --equation heat
```

Wave Equation

```bash
python train.py --equation wave
```

Burgers Equation

```bash
python train.py --equation burgers
```

Custom epochs

```bash
python train.py --equation heat --epochs 10000
```

Custom learning rate

```bash
python train.py --equation wave --lr 0.0005
```

---

## Evaluation

The framework reports:

- Relative L2 Error
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

It also generates:

- Prediction Plot
- Error Plot
- Heatmap
- 3D Surface Plot
- Training Loss Curve

---

## Technologies

- Python
- PyTorch
- NumPy
- Matplotlib

Machine Learning

- Physics-Informed Neural Networks (PINNs)
- Automatic Differentiation
- Scientific Machine Learning

---

## Future Work

- Finite Difference reference solvers
- Additional PDE support
- YAML configuration system
- TensorBoard integration
- Experiment logging

---

## References

Raissi, M., Perdikaris, P., & Karniadakis, G. E.

Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations.

PyTorch Documentation

---

## License

MIT License