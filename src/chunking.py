"""Chunking strategies."""

import ast
from pathlib import Path
from typing import List, Tuple


class Chunk:
    def __init__(
        self,
        file_path: str,
        text: str,
        first_character_index: int,
        last_character_index: int,
    ) -> None:
        self.file_path = file_path
        self.text = text
        self.first_character_index = first_character_index
        self.last_character_index = last_character_index


def _split_long_span(
    content: str,
    start: int,
    end: int,
    max_chunk_size: int,
) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    cursor = start

    while cursor < end:
        piece_end = min(cursor + max_chunk_size, end)

        if piece_end < end:
            newline_pos = content.rfind(chr(10), cursor, piece_end)
            if newline_pos > cursor:
                piece_end = newline_pos + 1

        spans.append((cursor, piece_end))
        cursor = piece_end

    return spans


def chunk_python_code(
    content: str,
    max_chunk_size: int,
) -> List[Tuple[int, int]]:
    if not content.strip():
        return []

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return _split_long_span(
            content,
            0,
            len(content),
            max_chunk_size,
        )

    lines = content.splitlines(keepends=True)

    line_offsets: List[int] = [0]

    for line in lines:
        line_offsets.append(line_offsets[-1] + len(line))

    top_level_nodes = [
        node
        for node in tree.body
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        )
    ]

    if not top_level_nodes:
        return _split_long_span(
            content,
            0,
            len(content),
            max_chunk_size,
        )

    boundaries: List[int] = [
        line_offsets[node.lineno - 1]
        for node in top_level_nodes
    ]

    spans: List[Tuple[int, int]] = []

    if boundaries[0] > 0:
        spans.extend(
            _split_long_span(
                content,
                0,
                boundaries[0],
                max_chunk_size,
            )
        )

    for idx, _node in enumerate(top_level_nodes):
        start_offset = boundaries[idx]

        end_offset = (
            boundaries[idx + 1]
            if idx + 1 < len(boundaries)
            else len(content)
        )

        spans.extend(
            _split_long_span(
                content,
                start_offset,
                end_offset,
                max_chunk_size,
            )
        )

    return [s for s in spans if s[1] > s[0]]


def chunk_text(
    content: str,
    max_chunk_size: int,
) -> List[Tuple[int, int]]:
    if not content.strip():
        return []

    paragraphs: List[Tuple[int, int]] = []
    cursor = 0
    length = len(content)

    while cursor < length:
        sep_idx = content.find(
            chr(10) + chr(10),
            cursor,
        )

        if sep_idx == -1:
            paragraphs.append((cursor, length))
            break

        end = sep_idx + 2
        paragraphs.append((cursor, end))
        cursor = end

    spans: List[Tuple[int, int]] = []
    current_start: int = -1
    current_end: int = -1

    for p_start, p_end in paragraphs:
        if p_end - p_start > max_chunk_size:
            if current_start != -1:
                spans.append((current_start, current_end))
                current_start = -1
                current_end = -1

            spans.extend(
                _split_long_span(
                    content,
                    p_start,
                    p_end,
                    max_chunk_size,
                )
            )
            continue

        if current_start == -1:
            current_start, current_end = p_start, p_end

        elif p_end - current_start <= max_chunk_size:
            current_end = p_end

        else:
            spans.append((current_start, current_end))
            current_start, current_end = p_start, p_end

    if current_start != -1:
        spans.append((current_start, current_end))

    return [s for s in spans if s[1] > s[0]]


def chunk_file(
    path: Path,
    content: str,
    max_chunk_size: int,
) -> List[Chunk]:
    if path.suffix == ".py":
        spans = chunk_python_code(
            content,
            max_chunk_size,
        )
    else:
        spans = chunk_text(
            content,
            max_chunk_size,
        )

    return [
        Chunk(
            file_path=str(path),
            text=content[start:end],
            first_character_index=start,
            last_character_index=end,
        )
        for start, end in spans
    ]
