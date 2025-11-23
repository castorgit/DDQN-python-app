"""Config loader for experiments: reads YAML and returns a dict of parameters."""
import yaml
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg
