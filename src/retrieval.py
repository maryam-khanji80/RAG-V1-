from pathlib import Path
from typing import Dict, List, cast

import bm25s

from .indexing import load_index
from .models import MinimalSource


class Retriever:
    """Wraps a BM25 index and chunk metadata for search."""

    def __init__(self, processed_dir: str, repo_root: str) -> None:
        """Initialize the retriever by loading the BM25 index from disk.

        Args:
            processed_dir: Directory containing 'bm25_index' and 'chunks'.
            repo_root: Root directory of the project.

        Raises:
            FileNotFoundError: If the index has not been built yet.
        """
        self.processed_dir = processed_dir
        self.repo_root = repo_root

        try:
            self._retriever: bm25s.BM25
            self._metadata: List[Dict[str, object]]
            self._retriever, self._metadata = load_index(processed_dir)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"No index found under '{processed_dir}'. "
                "Run 'python -m src index' first."
            ) from exc

    def search(self, query: str, k: int = 10) -> List[MinimalSource]:
        """Search the index for the top-k most relevant chunks.

        Args:
            query: Natural language query string.
            k: Number of results to return.

        Returns:
            List of MinimalSource objects ranked by BM25 relevance.
        """
        if not query or not query.strip():
            return []

        effective_k = max(1, min(k, len(self._metadata)))
        query_tokens = bm25s.tokenize(
            [query], stopwords="en", show_progress=False
        )

        results, _scores = self._retriever.retrieve(
            query_tokens, k=effective_k, show_progress=False
        )

        sources: List[MinimalSource] = []

        for idx in results[0]:
            meta = self._metadata[int(idx)]

            sources.append(
                MinimalSource(
                    file_path=str(meta["file_path"]),
                    first_character_index=cast(
                        int,
                        meta["first_character_index"],
                    ),
                    last_character_index=cast(
                        int,
                        meta["last_character_index"],
                    ),
                )
            )

        return sources

    def get_chunk_text(self, source: MinimalSource) -> str:
        """Read the text of a chunk from disk using its stored offsets.

        Args:
            source: A MinimalSource with file_path and character offsets.

        Returns:
            The text slice, or empty string if the file cannot be read.
        """
        try:
            file_path = Path(source.file_path)

            if not file_path.is_absolute() and not file_path.exists():
                candidate = Path(self.repo_root) / source.file_path

                if candidate.exists():
                    file_path = candidate

            content = file_path.read_text(
                encoding="utf-8",
                errors="ignore",
            )

            return content[
                source.first_character_index: source.last_character_index
            ]

        except OSError:
            return ""
