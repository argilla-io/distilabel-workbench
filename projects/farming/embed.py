from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")


def embed_dataset(dataset: Dataset, columns: list):

    def _embed(sample):
        return {f"{col}_vector": model.encode(sample[col]) for col in columns}

    dataset = dataset.map(_embed)
    return dataset


if __name__ == "__main__":
    from argparse import ArgumentParser

    default_repo_id = "argilla/farming"
    default_dataset_split = "train"

    parser = ArgumentParser()
    parser.add_argument("--repo_id", type=str, default=default_repo_id)
    parser.add_argument("--split", type=str, default=default_dataset_split)
    parser.add_argument(
        "--columns",
        type=list,
        nargs="+",
        default=["instruction", "answer", "rationale", "task"],
    )
    args = parser.parse_args()

    dataset = load_dataset(args.repo_id, split=args.split)
    dataset = embed_dataset(dataset, columns=args.columns)
    dataset.push_to_hub(args.repo_id)
