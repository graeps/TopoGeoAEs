def is_valid_model_config(config):
    """
    Verify if the input config dictionary is a valid model configuration.

    Args:
        config (dict): The dictionary to be verified.

    Returns:
        bool: True if the config is valid, False otherwise.
    """
    required_keys = {
        'data_dim': int,
        'latent_dim': int,
        'sftbeta': (int, float),
        'device': str,
        'encoder_width': int,
        'encoder_depth': int,
        'decoder_width': int,
        'decoder_depth': int
    }

    for key, expected_type in required_keys.items():
        if key not in config:
            raise InvalidModelConfigError(f"Missing key: {key}")
        if not isinstance(config[key], expected_type):
            raise InvalidModelConfigError(
                f"Invalid type for key: {key}. Expected {expected_type}, got {type(config[key])}.")

    return True


class InvalidModelConfigError(Exception):
    """Custom exception to indicate an invalid model configuration."""
    pass
