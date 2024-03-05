import gzip
import io
import json
import re
from collections import defaultdict

import requests
from datasets import Dataset, load_dataset
from datasets.dataset_dict import DatasetDict, IterableDatasetDict
from datasets.iterable_dataset import IterableDataset

pattern = re.compile(r"\n\nHuman:(.*?)\n\nAssistant:", re.DOTALL)


def _shuffle_and_select(ds, n):
    return ds.shuffle().select(range(n))


def _get_jsonl_from_github(url):
    response = requests.get(url, stream=True)
    dict_list = defaultdict(list)
    if response.status_code == 200:
        # Create an io.BytesIO object and write the content directly into it
        byte_stream = io.BytesIO()
        for chunk in response.iter_content(chunk_size=128):
            byte_stream.write(chunk)

        # Reset the stream position to the beginning
        byte_stream.seek(0)

        # Use gzip to decompress the io.BytesIO stream
        with gzip.GzipFile(fileobj=byte_stream, mode="rb") as decompressed_stream:
            # Load JSON content from the decompressed io.BytesIO stream
            for line in decompressed_stream.readlines():
                print(line)
                exit()
                try:
                    dict_for_list = json.loads(line)
                    for k, v in dict_for_list.items():
                        dict_list[k].append(v)
                except json.JSONDecodeError as e:
                    print(f"Failed to decode JSON: {e}")

    else:
        print(f"Failed to download the file. Status code: {response.status_code}")

    ds = Dataset.from_dict(dict_list)
    return ds


def load_self_instruct():
    ds = load_dataset("yizhongw/self_instruct", "human_eval")
    ds = ds["train"]
    ds = _shuffle_and_select(ds, 100)
    ds = ds.map(
        lambda x: {
            "instruction": x["instruction"]
            + " "
            + "\n".join(x["instances"].get("input"))
        }
    )
    ds = ds.rename_column("instruction", "input")

    return ds


def load_koala():
    ds = load_dataset("HuggingFaceH4/Koala-test-set")
    ds = ds["test"]
    ds = _shuffle_and_select(ds, 100)
    ds = ds.rename_column("prompt", "input")
    return ds


def load_vicuna():
    ds = load_dataset("zhengxuanzenwu/vicuna-eval-with-gpt4")
    ds = _shuffle_and_select(ds, 80)
    ds = ds["test"]
    ds = ds.rename_column("instruction", "input")
    return ds


def load_anthropic_red_team_attempts():
    # url = "https://github.com/anthropics/hh-rlhf/raw/master/red-team-attempts/red_team_attempts.jsonl.gz"
    # ds = _get_jsonl_from_github(url)
    # ds = _shuffle_and_select(ds, 100)
    # print(ds)
    # ds = ds.map(lambda x: {"chosen": pattern.search(x["chosen"]).group(1).strip()})
    # ds = ds.rename_column("chosen", "input")
    ds: DatasetDict | Dataset | IterableDatasetDict | IterableDataset = load_dataset(
        "Anthropic/hh-rlhf", data_dir="red-team-attempts"
    )
    ds = ds["train"]
    ds = _shuffle_and_select(ds, 100)
    ds = ds.map(
        lambda x: {"transcript": pattern.search(x["transcript"]).group(1).strip()}
    )
    ds = ds.rename_column("transcript", "input")

    return ds


def load_anthropic_harmless_base():
    ds: DatasetDict | Dataset | IterableDatasetDict | IterableDataset = load_dataset(
        "Anthropic/hh-rlhf", data_dir="harmless-base"
    )
    ds = ds["test"]
    ds = _shuffle_and_select(ds, 100)
    ds = ds.map(lambda x: {"chosen": pattern.search(x["chosen"]).group(1).strip()})
    ds = ds.rename_column("chosen", "input")

    return ds


def load_best_chatgpt_prompts():
    ds = load_dataset("yizhongw/best_chatgpt_prompts")
    ds = _shuffle_and_select(ds, 100)
    return ds


def load_oasst1():
    ds = load_dataset("OpenAssistant/oasst1")
    ds = ds["validation"]
    ds = ds.filter(lambda x: x["lang"] == "en")
    ds = _shuffle_and_select(ds, 100)
    ds = ds.rename_column("text", "input")

    return


# load_self_instruct()
# load_koala()
# load_vicuna()
# load_anthropic_red_team_attempts()
# load_anthropic_harmless_base()
# load_best_chatgpt_prompts()
load_oasst1()
