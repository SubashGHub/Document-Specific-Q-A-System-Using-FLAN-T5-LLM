import yaml
import os

filepath = os.path.join(os.path.dirname(__file__), 'config.yaml')

with open(filepath, "r") as f:
    config = yaml.safe_load(f)

def get_path(config_name):
    return config[config_name]