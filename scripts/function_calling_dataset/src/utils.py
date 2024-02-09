from datetime import datetime
from pathlib import Path
from uuid import uuid4

from distilabel.dataset import DatasetCheckpoint, Dataset
import yaml


def load_wrapped_dataset(dataset_path: str, unwrap: callable = lambda x: x):
    functions_dataset = Dataset.load_from_disk(dataset_path)
    functions_dataset = unwrap(functions_dataset)
    return functions_dataset


def setup_checkpoint_strategy(data_dir: str, checkpoint_name: str):
    path = Path(data_dir) / "checkpoints" / checkpoint_name
    config_path = Path(data_dir) / "config.yaml"
    save_frequency = load_config(str(config_path)).get("save_frequency", -1)
    checkpoint_strategy = DatasetCheckpoint(
        path=path,
        save_frequency=save_frequency,
    )
    return checkpoint_strategy


def setup_run(config_path: str):
    config = load_config(config_path)
    dataset_name = config.get("name", f"function-calling-dataset-{uuid4()}")
    data_dir = (
        Path("data") / dataset_name / datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    )
    data_dir.mkdir(exist_ok=True)
    config["data_dir"] = str(data_dir)
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
