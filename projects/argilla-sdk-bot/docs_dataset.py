r"""Script to generate a dataset from the markdown docs files in a repository.

Usage:
$ python docs_dataset.py -h

Example:
$ python docs_dataset.py \
    "argilla-io/argilla-python" \
    "my-name/argilla_sdk_docs_raw" \
    --max_chars 512
"""

import pandas as pd
from datasets import Dataset

from github import Github, Repository, ContentFile
import requests
import os

from typing import List

import nltk

from pathlib import Path
from markdown_it import MarkdownIt
from markdown_it.token import Token


# The github related functions are a copy from the following repository
# https://github.com/Nordgaren/Github-Folder-Downloader/blob/master/gitdl.py
def download(c: ContentFile, out: str) -> None:
    r = requests.get(c.download_url)
    output_path = f"{out}/{c.path}"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        print(f"downloading {c.path} to {out}")
        f.write(r.content)


def download_folder(repo: Repository, folder: str, out: str, recursive: bool) -> None:
    contents = repo.get_contents(folder)
    for c in contents:
        if c.download_url is None:
            if recursive:
                download_folder(repo, c.path, out, recursive)
            continue
        download(c, out)


def read_file(filename: Path) -> str:
    """Read a whole markdown file to a string."""
    with open(filename, "r") as f:
        return f.read()


## Some helper functions to find specific pieces in the markdown files
## that we may want to set aside.


def is_front_matter(text: str) -> bool:
    """Check if a token pertains to the front matter.

    The check seeks if the string starts with '---' and
    the word `title` after a single line jump (it will fail if some
    space is inserted between them), and ends with '---'.

    Args:
        text (str):
            text obtained in the Token's content.
            Expects to be applied to the tokens from a markdown parsed.

    Returns:
        bool
    """
    return text.startswith("---\n") and text.endswith("\n---")


def is_figure(text: str) -> bool:
    """Check if a paragraph is just a picture in the doc.

    Some lines may contain just a picture, and there is no
    reason to translate those.
    i.e.
    '![helpner](/images/helpner-arch-part1.png)'
    The type of check is not perfect, it just fits my needs.

    Args:
        text (str): text obtained in the Token's content.

    Returns:
        bool:
    """
    text = text.strip()
    return text.startswith("![") and text.endswith(")")


def is_code(text: str) -> bool:
    """Check if a blob of text is a chunk of code.

    Args:
        text (str): text obtained in the Token's content.

    Returns:
        bool
    """
    text = text.strip()
    return text.startswith("```") and text.endswith("```")


def is_comment(text: str) -> bool:
    """Check if a blob of text is a comment."""
    text = text.strip()
    return text.startswith("<!--") and text.endswith("-->")


def get_text_pieces(md_tokens: List[Token]) -> List[str]:
    """Obtains the text pieces from the markdown tokens.

    Parses the markdown file to obtain the text as extracted by the parser

    Args:
        md_tokens: List of tokens obtained from the markdown parser.

    Returns:
        List of strings that can be used to generate the chunks.
    """
    text_pieces = []
    for t in md_tokens:
        if t.type == "inline":
            if any(
                (
                    is_front_matter(t.content),
                    is_figure(t.content),
                    # is_code(t.content),
                    is_comment(t.content),
                )
            ):
                continue
            text_pieces.append(t.content)
    return text_pieces


def chunk_texts(text_pieces: List[str], max_chars: int = 512) -> List[str]:
    """Function to generate the chunks of text from the text pieces.

    It loops over the text pieces to generate the chunks of text of roughly
    the same length given by the `max_chars` parameter.

    Note:
        The algorithm is not perfect, it may generate chunks that are of different
        sizes, and pieces of code may be split in the middle, it can be further improved.

    Args:
        text_pieces: List of strings to be chunked, the strings from the markdown file.
        max_chars: It will determine the approximate size of the chunks
            in RAG. Defaults to 512.

    Returns:
        List of strings chunked.
    """
    chunks = []
    current_chunk = []
    current_chars = 0

    for text in text_pieces:
        if len(text) > max_chars:
            sentences = nltk.sent_tokenize(text)
            if len(sentences) > 1:
                # To avoid infinite recursion
                pieces = chunk_texts(sentences, max_chars)
                chunks.extend(pieces)
            else:
                current_chunk.extend(sentences)
                chunks.extend(sentences)

        elif current_chars + len(text) > max_chars:
            chunks.append(". ".join(current_chunk))
            current_chunk = [text]
            current_chars = len(text)

        else:
            current_chunk.append(text)
            current_chars += len(text)

    if current_chunk:
        chunks.append(". ".join(current_chunk))

    return chunks


def create_chunks(md_files: List[Path], max_chars: int = 512) -> dict[str, List[str]]:
    """Create the chunks of text from the markdown files.

    Args:
        md_files: List of paths to the markdown files.
        max_chars: The approximate maximum number of characters for each chunk
            obtained from a file. Defaults to 512.

    Returns:
        Dictionary from filename to the list of chunks.
    """
    md = MarkdownIt("zero")
    data = {}
    for file in md_files:
        contents = read_file(file)
        md_tokens = md.parse(contents)
        text_pieces = get_text_pieces(md_tokens)
        chunks = chunk_texts(text_pieces, max_chars=max_chars)
        data[str(file)] = chunks
    return data


def create_dataset(data: dict[str, List[str]]) -> Dataset:
    """Creates a dataset from the dictionary of chunks.

    Args:
        data: Dictionary from filename to the list of chunks,
            as obtained from `create_chunks`.

    Returns:
        Dataset with `filename` and `chunks` columns.
    """
    df = pd.DataFrame.from_records(
        [(k, v) for k, values in data.items() for v in values],
        columns=["filename", "chunks"],
    )
    ds = Dataset.from_pandas(df)
    return ds


def main():
    import argparse

    description = (
        "Download the docs from a github repository and generate a dataset "
        "from the markdown files. The dataset will be pushed to the hub."
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "repo",
        help="Name of the repository in the hub. For example 'argilla-io/argilla-python'.",
    )
    parser.add_argument(
        "dataset_name",
        help="Name to give to the new dataset. For example 'my-name/argilla_sdk_docs_raw'.",
    )
    parser.add_argument(
        "--docs_folder",
        default="docs",
        help="Name of the docs folder in the repo, defaults to 'docs'.",
    )
    parser.add_argument(
        "--max_chars",
        type=int,
        default=256,
        help="Maximum number of characters for each chunk. Defaults to 256.",
    )
    parser.add_argument(
        "--output_dir",
        help="Path to save the downloaded files from the repo (optional)",
    )
    parser.add_argument(
        "--private",
        action=argparse.BooleanOptionalAction,
        help="Whether to keep the repository private or not. Defaults to False.",
    )

    args = parser.parse_args()

    # Instantiate the Github object to download the files
    print("Instantiate repository...")
    gh = Github()
    repo = gh.get_repo(args.repo)

    docs_path = Path(args.output_dir or args.repo.split("/")[1])

    if docs_path.exists():
        print(f"Folder {docs_path} already exists, skipping download.")
    else:
        print("Start downloading the files...")
        download_folder(repo, args.docs_folder, str(docs_path), True)

    md_files = list(docs_path.glob("**/*.md"))

    # Loop to iterate over the files and generate chunks from the text pieces
    print("Generating the chunks from the markdown files...")
    data = create_chunks(md_files, max_chars=args.max_chars)

    # Create a dataset to push it to the hub
    print("Creating the dataset...")
    ds = create_dataset(data)
    ds.push_to_hub(args.dataset_name, private=args.private)
    print("Dataset pushed to the hub")


if __name__ == "__main__":
    # Download the punkt tokenizer, will need it for the sentence tokenizer
    nltk.download("punkt")
    main()
