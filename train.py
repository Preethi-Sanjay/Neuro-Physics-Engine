import argparse
import torch
import torch.optim as optim

from models.pinn import PINN
from utils.equation_loader import load_equation
from utils.sampling import sample_collocation_points

from losses.physics_loss import physics_loss
from losses.initial_loss import (
    initial_condition_loss,
    initial_velocity_loss,
)
from losses.boundary_loss import boundary_condition_loss

from evaluate import evaluate


# ==========================================
# Command Line Arguments
# ==========================================

parser = argparse.ArgumentParser(
    description="NeuroPhysicsEngine"
)

parser.add_argument(
    "--equation",
    type=str,
    default="heat",
    choices=[
        "heat",
        "wave",
        "burgers",
    ],
    help="Equation to solve",
)

parser.add_argument(
    "--epochs",
    type=int,
    default=5000,
)

parser.add_argument(
    "--lr",
    type=float,
    default=1e-3,
)

parser.add_argument(
    "--points",
    type=int,
    default=1000,
    help="Number of collocation points",
)

args = parser.parse_args()


# ==========================================
# Configuration
# ==========================================

EQUATION = args.equation
EPOCHS = args.epochs
LEARNING_RATE = args.lr
COLLOCATION_POINTS = args.points


# ==========================================
# Device
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# Load Equation
# ==========================================

equation = load_equation(EQUATION)
residual_function = equation.residual


# ==========================================
# Model
# ==========================================

model = PINN().to(device)


# ==========================================
# Optimizer
# ==========================================

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ==========================================
# Training
# ==========================================

loss_history = []

print("=" * 50)
print(f"Training {EQUATION.upper()} Equation")
print("=" * 50)

for epoch in range(EPOCHS):

    optimizer.zero_grad()

    x, t = sample_collocation_points(
        COLLOCATION_POINTS
    )

    x = x.to(device)
    t = t.to(device)

    residual = residual_function(
        model,
        x,
        t,
    )

    physics = physics_loss(residual)

    ic = initial_condition_loss(
        model,
        x,
        equation,
    )

    iv = initial_velocity_loss(
        model,
        x,
        equation,
    )

    bc = boundary_condition_loss(
        model,
        t,
        equation,
    )

    loss = physics + ic + iv + bc

    loss.backward()

    optimizer.step()

    loss_history.append(loss.item())

    if epoch % 500 == 0:

        print(
            f"Epoch {epoch:5d} | "
            f"Physics: {physics.item():.6f} | "
            f"IC: {ic.item():.6f} | "
            f"IV: {iv.item():.6f} | "
            f"BC: {bc.item():.6f} | "
            f"Total: {loss.item():.6f}"
        )


# ==========================================
# Save Model
# ==========================================

import os

os.makedirs(
    "checkpoints",
    exist_ok=True,
)

checkpoint = f"checkpoints/{EQUATION}_pinn.pth"

torch.save(
    model.state_dict(),
    checkpoint,
)

print("\nModel saved successfully!")
print(f"Checkpoint: {checkpoint}")


# ==========================================
# Evaluate
# ==========================================

evaluate(
    model,
    loss_history,
    equation,
)