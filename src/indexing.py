import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import bm25s
from tqdm import tqdm

from .chunking import Chunk, chunk_file


INCLUDED_SUFFIXES = {".py", ".md", ".txt"}

EXCLUDED_DIR_PARTS = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".venv",
    "node_modules",
    ".tox",
    "build",
    "dist",
    "eggs",
    ".eggs",
}


def _iter_repo_files(repo_path: Path) -> List[Path]:
    files: List[Path] = []

    for path in repo_path.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix not in INCLUDED_SUFFIXES:
            continue

        if any(
            part in EXCLUDED_DIR_PARTS
            for part in path.parts
        ):
            continue

        files.append(path)

    return sorted(files)


def build_chunks(
    repo_path: str,
    max_chunk_size: int,
) -> List[Chunk]:
    path_object = Path(repo_path)
    chunks: List[Chunk] = []
    file_list = _iter_repo_files(path_object)

    for file_path in tqdm(
        file_list,
        desc="Chunking files",
    ):
        try:
            content = file_path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except OSError:
            continue

        if not content:
            continue

        file_chunks = chunk_file(
            file_path,
            content,
            max_chunk_size,
        )

        chunks.extend(file_chunks)

    return chunks


def save_index(
    chunks: List[Chunk],
    output_dir: str,
) -> None:
    output_path = Path(output_dir)
    bm25_dir = output_path / "bm25_index"
    chunks_dir = output_path / "chunks"

    bm25_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    chunks_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    corpus = [
        chunk.text
        for chunk in chunks
    ]

    print(f"Tokenizing {len(corpus)} chunks...")

    tokenized = bm25s.tokenize(
        corpus,
        stopwords="english",
        show_progress=True,
    )

    print("Building BM25 index...")

    retriever = bm25s.BM25()
    retriever.index(
        tokenized,
        show_progress=True,
    )

    retriever.save(str(bm25_dir))

    metadata = [
        {
            "file_path": chunk.file_path,
            "first_character_index": chunk.first_character_index,
            "last_character_index": chunk.last_character_index,
        }
        for chunk in chunks
    ]

    with open(
        chunks_dir / "chunks.pkl",
        "wb",
    ) as f:
        pickle.dump(metadata, f)

    with open(
        chunks_dir / "chunks.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print(f"Indexed {len(chunks)} chunks.")


def load_index(
    processed_dir: str,
) -> Tuple[bm25s.BM25, List[Dict[str, object]]]:
    processed = Path(processed_dir)
    bm25_path = processed / "bm25_index"
    chunks_path = processed / "chunks" / "chunks.pkl"

    if not bm25_path.exists():
        raise FileNotFoundError(
            f"BM25 index not found at '{bm25_path}'."
        )

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunk metadata not found at '{chunks_path}'."
        )

    retriever = bm25s.BM25.load(
        str(bm25_path),
        load_corpus=False,
    )

    with open(chunks_path, "rb") as f:
        metadata: List[Dict[str, object]] = pickle.load(f)

    return retriever, metadata
