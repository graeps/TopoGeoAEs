import os
from types import SimpleNamespace


def _describe_experiment(overrides):
    """
    Generates a concise string description of an experiment configuration by summarizing
    all key-value overrides except the "experiment" key.

    Args:
        overrides (dict): A dictionary containing configuration key-value
            pairs. All key-value pairs except those with the key "experiment"
            are included in the generated description.

    Returns:
        str: A string representation summarizing the key-value pairs
            from the `overrides` dictionary, excluding the "experiment" key.

    """
    desc_lines = []
    for k, v in overrides.items():
        if k != "experiment":
            desc_lines.append(f"{k}={v}")
    return ", ".join(desc_lines)


def generate_experiments(base_configuration, parameter_grid):
    """
    Generates a collection of experimental configurations by applying a parameter grid to a base configuration.
    This function combines a template configuration (`base_configuration`) with values provided in a
    parameter grid (`parameter_grid`) to generate a series of experiments. Each experiment is identified by
    a unique name and the function ensures that all parameters in the grid are synchronized for iteration.

    Args:
        base_configuration (dict): The base experimental configuration to be overridden by parameter grid values.
        parameter_grid (dict): A dictionary where keys represent parameter names, and values are lists of parameter
            values. All value lists must have the same length for synchronized configuration creation.

    Raises:
        ValueError: If the parameter list lengths in `parameter_grid` are not all the same.

    Returns:
        dict: A dictionary of experiment configurations, where each key is a unique experiment name and each
            value is a `SimpleNamespace` containing the complete configuration for that experiment.
    """
    # Ensure all lists are of equal length
    lengths = [len(v) for v in parameter_grid.values()]
    if len(set(lengths)) != 1:
        raise ValueError("All parameter lists in param_grid must have the same length for synchronized iteration.")

    n = lengths[0]
    experiments = {}

    for i in range(n):
        overrides = {k: v[i] for k, v in parameter_grid.items() if v[i] != "_"}

        base_name = base_configuration.get("experiment", "default")
        name = f"exp{i:02d}_{base_name}"  # concise ID with base experiment name

        overrides["experiment"] = name

        cfg = base_configuration.copy()
        cfg.update(overrides)
        cfg["description"] = _describe_experiment(overrides)

        if cfg.get("logging"):
            default_root_log_dir = "./results"
            if cfg.get("log_dir") is None:
                log_dir = os.path.join(default_root_log_dir, cfg["model_type"], cfg["dataset_name"], f"results_{name}")
            else:
                log_dir = os.path.join(cfg["log_dir"], cfg["model_type"], f"results_{name}")
            os.makedirs(log_dir, exist_ok=True)
            cfg["log_dir"] = log_dir
        else: cfg["log_dir"] = None

        experiments[name] = SimpleNamespace(**cfg)

    return experiments
