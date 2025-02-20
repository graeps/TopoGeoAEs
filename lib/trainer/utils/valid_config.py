def is_valid_trainer_config(config):
    """
    Verify if the input config dictionary is a valid model configuration.

    Args:
        config (dict): The dictionary to be verified.

    Returns:
        bool: True if the config is valid, False otherwise.
    """
    required_keys = {
        'num_epochs': int,
        'log_interval': int,
        'device': str,
        'recon_loss': str,
    }

    for key, expected_type in required_keys.items():
        if key not in config:
            raise InvalidTrainerConfigError(f"Missing key: {key}")
        if not isinstance(config[key], expected_type):
            raise InvalidTrainerConfigError(
                f"Invalid type for key: {key}. Expected {expected_type}, got {type(config[key])}.")

    return True


class InvalidTrainerConfigError(Exception):
    """Custom exception to indicate an invalid model configuration."""
    pass
