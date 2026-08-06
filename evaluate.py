import torch
import matplotlib.pyplot as plt

from utils.metrics import (
    relative_l2_error,
    mae,
    mse,
    rmse,
)

from utils.visualization import (
    plot_prediction,
    plot_error,
    plot_heatmap,
    plot_surface,
    plot_loss,
)


def evaluate(model, loss_history, equation_module):
    """
    Evaluate a trained PINN model.

    Required functions inside equation module:

        residual()
        analytical_solution()
        initial_condition()
        boundary_condition_left()
        boundary_condition_right()

    """

    device = next(model.parameters()).device

    model.eval()

    print("\n==============================")
    print("     Evaluation Report")
    print("==============================")

    # ==========================================
    # 1D Evaluation
    # ==========================================

    x = torch.linspace(0, 1, 200).reshape(-1, 1).to(device)

    t = torch.full_like(x, 0.5)

    with torch.no_grad():

        inputs = torch.cat((x, t), dim=1)

        prediction = model(inputs)

        target = equation_module.analytical_solution(
            x,
            t,
        )

    # ==========================================
    # Metrics
    # ==========================================

    l2 = relative_l2_error(
        prediction,
        target,
    )

    mae_error = mae(
        prediction,
        target,
    )

    mse_error = mse(
        prediction,
        target,
    )

    rmse_error = rmse(
        prediction,
        target,
    )

    print(f"\nRelative L2 Error : {l2:.6f}")
    print(f"MAE               : {mae_error:.6f}")
    print(f"MSE               : {mse_error:.6f}")
    print(f"RMSE              : {rmse_error:.6f}")

    # ==========================================
    # Prediction Plot
    # ==========================================

    print("\nCreating Prediction Plot...")

    plot_prediction(
        x.cpu(),
        prediction.cpu(),
        target.cpu(),
    )

    # ==========================================
    # Error Plot
    # ==========================================

    print("Creating Error Plot...")

    plot_error(
        x.cpu(),
        prediction.cpu(),
        target.cpu(),
    )

    # ==========================================
    # 2D Evaluation
    # ==========================================

    print("Creating Solution Grid...")

    grid = 100

    x_grid = torch.linspace(
        0,
        1,
        grid,
    )

    t_grid = torch.linspace(
        0,
        1,
        grid,
    )

    X, T = torch.meshgrid(
        x_grid,
        t_grid,
        indexing="ij",
    )

    inputs = torch.cat(
        (
            X.reshape(-1, 1),
            T.reshape(-1, 1),
        ),
        dim=1,
    ).to(device)

    with torch.no_grad():

        prediction2 = model(inputs)

    U = prediction2.cpu().numpy().reshape(
        grid,
        grid,
    )

    # ==========================================
    # Heatmap
    # ==========================================

    print("Creating Heatmap...")

    plot_heatmap(
        X.numpy(),
        T.numpy(),
        U,
    )

    # ==========================================
    # Surface Plot
    # ==========================================

    print("Creating Surface Plot...")

    plot_surface(
        X.numpy(),
        T.numpy(),
        U,
    )

    # ==========================================
    # Loss Plot
    # ==========================================

    print("Creating Loss Plot...")

    plot_loss(loss_history)

    # ==========================================
    # Display
    # ==========================================

    plt.show()

    print("\n==============================")
    print(" Evaluation Completed")
    print("==============================")