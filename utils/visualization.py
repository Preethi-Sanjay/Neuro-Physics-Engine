import matplotlib.pyplot as plt
import torch


# ==========================
# Dashboard Figure
# ==========================

fig, axs = plt.subplots(
    2,
    2,
    figsize=(15, 10),
    num="PINN Dashboard"
)


# ==========================
# Prediction Plot
# ==========================

def plot_prediction(x, prediction, target):

    x = x.detach().cpu().numpy().squeeze()
    prediction = prediction.detach().cpu().numpy().squeeze()
    target = target.detach().cpu().numpy().squeeze()

    axs[0, 0].clear()

    axs[0, 0].plot(
        x,
        target,
        linewidth=2,
        label="Analytical Solution"
    )

    axs[0, 0].plot(
        x,
        prediction,
        "--",
        linewidth=2,
        label="PINN Prediction"
    )

    axs[0, 0].set_title("PINN vs Analytical Solution")
    axs[0, 0].set_xlabel("Position (x)")
    axs[0, 0].set_ylabel("Temperature")
    axs[0, 0].legend()
    axs[0, 0].grid(True)


# ==========================
# Error Plot
# ==========================

def plot_error(x, prediction, target):

    x = x.detach().cpu().numpy().squeeze()
    error = torch.abs(prediction - target).detach().cpu().numpy().squeeze()

    axs[0, 1].clear()

    axs[0, 1].plot(
        x,
        error,
        color="red"
    )

    axs[0, 1].set_title("Absolute Error")
    axs[0, 1].set_xlabel("Position (x)")
    axs[0, 1].set_ylabel("Error")
    axs[0, 1].grid(True)


# ==========================
# Heatmap
# ==========================

def plot_heatmap(X, T, U):

    axs[1, 0].clear()

    image = axs[1, 0].imshow(
        U,
        extent=[0, 1, 0, 1],
        origin="lower",
        aspect="auto",
        cmap="hot"
    )

    axs[1, 0].set_title("Temperature Heatmap")
    axs[1, 0].set_xlabel("Position (x)")
    axs[1, 0].set_ylabel("Time (t)")

    fig.colorbar(
        image,
        ax=axs[1, 0]
    )


# ==========================
# Loss Curve
# ==========================

def plot_loss(loss_history):

    axs[1, 1].clear()

    axs[1, 1].plot(
        loss_history,
        color="green"
    )

    axs[1, 1].set_title("Training Loss")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("Loss")
    axs[1, 1].grid(True)

    fig.suptitle(
        "Physics-Informed Neural Network Results",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()


# ==========================
# 3D Surface Plot
# ==========================

def plot_surface(X, T, U):

    fig3d = plt.figure(
        figsize=(10, 7),
        num="3D Temperature Surface"
    )

    ax = fig3d.add_subplot(
        111,
        projection="3d"
    )

    ax.plot_surface(
        X,
        T,
        U,
        cmap="viridis"
    )

    ax.set_title("PINN Solution Surface")
    ax.set_xlabel("Position (x)")
    ax.set_ylabel("Time (t)")
    ax.set_zlabel("Temperature")

    plt.tight_layout()