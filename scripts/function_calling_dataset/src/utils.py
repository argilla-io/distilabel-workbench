from datetime import datetime
from pathlib import Path
from uuid import uuid4

import yaml


def setup_run(config_path: str):
    config = load_config(config_path)
    dataset_name = config.get("name", f"function-calling-dataset-{uuid4()}")
    data_dir = (
        Path("data") / dataset_name / datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    )
    data_dir.mkdir(exist_ok=True)
    dump_config(config, data_dir / "config.yaml")
    return data_dir, dataset_name, config


def dump_config(config, config_path: str):
    with open(config_path, "w") as file:
        yaml.dump(config, file)


def load_config(config_path: str):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def filter_column_not_none(dataset, column):
    return dataset.filter(lambda x: x[column] is not None)
