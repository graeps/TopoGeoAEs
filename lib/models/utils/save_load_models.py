import json
import os
import time
import torch

from .. import EuclideanVAE
from .. import ToroidalVAE
from .valid_config import is_valid_model_config

path_to_pretrained = "./pretrained_models/"


def get_model(posterior_type):
    model_map = {
        "gaussian": EuclideanVAE,
        "toroidal": ToroidalVAE,
    }
    return model_map.get(posterior_type.lower(), None)  # None if model_type is not found


def save_model(model, model_config, name='', save_dir=path_to_pretrained):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = str(int(time.time()))

    model_name = model.posterior_type + "_" + name + f'{timestamp}.pth'
    model_path = os.path.join(save_dir, model_name)

    # save model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as {model_name}")

    # save config
    config_name = model.posterior_type + "_" + name + f'{timestamp}.json'
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
