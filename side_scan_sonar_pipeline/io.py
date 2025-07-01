import pandas as pd
import yaml

def load_csv(filepath):
    return pd.read_csv(filepath)

def load_yaml_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
