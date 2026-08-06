import importlib


SUPPORTED_EQUATIONS = {
    "heat": "equations.heat",
    "wave": "equations.wave",
    "burgers": "equations.burgers",
    "laplace": "equations.laplace",
    "poisson": "equations.poisson",
}


def load_equation(name: str):
    """
    Dynamically loads an equation module.

    Example:
        equation = load_equation("heat")
    """

    name = name.lower()

    if name not in SUPPORTED_EQUATIONS:
        raise ValueError(
            f"Unsupported equation '{name}'.\n"
            f"Supported equations: {list(SUPPORTED_EQUATIONS.keys())}"
        )

    return importlib.import_module(
        SUPPORTED_EQUATIONS[name]
    )