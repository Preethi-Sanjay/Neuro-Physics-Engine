import yaml


def load_config(path):
    """
    Load a YAML configuration file.
    """

    with open(path, "r") as file:
        config = yaml.safe_load(file)

    return config