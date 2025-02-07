import json
import os
import time
import torch

from ..models import EuclideanVAE, ToroidalVAE

path_to_pretrained = "../pretrained_models/"


def get_model(posterior_type):
    model_map = {
        "gaussian": EuclideanVAE,
        "toroidal": ToroidalVAE,
    }
    return model_map.get(posterior_type.lower(), None)  # None if model_type is not found


def save_model(model, model_config, save_dir=path_to_pretrained):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = str(int(time.time()))

    model_name = model.posterior_type + "_" + f'{timestamp}.pth'
    model_path = os.path.join(save_dir, model_name)

    # save model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as {model_name}")

    # save config
    config_name = model.posterior_type + "_" + f'{timestamp}.json'
    config_path = os.path.join(save_dir, config_name)
    with open(config_path, 'w') as file:
        json.dump(model_config, file)

    print(f"Model saved as {config_name}")


def load_model(model_file_name, save_dir=path_to_pretrained):
    posterior_type = model_file_name.split('_')[0]
    model_path = os.path.join(save_dir, model_file_name + '.pth')
    config_path = os.path.join(save_dir, model_file_name + '.json')

    # Load config
    with open(config_path, 'r') as file:
        model_config = json.load(file)

    is_valid_model_config(model_config)

    model = get_model(posterior_type)(model_config)
    model.load_state_dict(
        torch.load(model_path, weights_only=True))
    model.eval()

    print(f"Model loaded from {model_path}")

    return model


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
