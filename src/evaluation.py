from typing import List, Dict

from .models import MinimalSource


def _overlap_ratio(
    retrieved: MinimalSource,
    correct: MinimalSource,
) -> float:

    if retrieved.file_path != correct.file_path:
        return 0.0

    start = max(
        retrieved.first_character_index,
        correct.first_character_index,
    )
    end = min(
        retrieved.last_character_index,
        correct.last_character_index,
    )
    overlap = max(0, end - start)

    correct_len = (
        correct.last_character_index
        - correct.first_character_index
    )

    if correct_len <= 0:
        return 0.0

    return overlap / correct_len


def is_source_found(
    retrieved: List[MinimalSource],
    correct: MinimalSource,
    threshold: float = 0.05,
) -> bool:
    return any(
        _overlap_ratio(r, correct) >= threshold
        for r in retrieved
    )


def recall_at_k(
    retrieved: List[MinimalSource],
    correct_sources: List[MinimalSource],
    k: int,
) -> float:
    if not correct_sources:
        return 0.0

    top_k_retrieved = retrieved[:k]

    found = sum(
        1
        for c in correct_sources
        if is_source_found(top_k_retrieved, c)
    )

    return found / len(correct_sources)


def evaluate_dataset(
    src_results: List[Dict[str, object]],
    ground_truth: Dict[str, Dict[str, object]],
    ks: List[int],
) -> Dict[str, float]:
    scores: Dict[int, List[float]] = {k: [] for k in ks}

    for result in src_results:
        qid = str(result.get("question_id", ""))

        if qid not in ground_truth:
            continue

        retrieved_raw = result.get("retrieved_sources", [])
        correct_raw = ground_truth[qid].get("sources", [])

        if not isinstance(retrieved_raw, list):
            retrieved_raw = []

        if not isinstance(correct_raw, list):
            correct_raw = []

        retrieved = [
            MinimalSource(**s) if isinstance(s, dict) else s
            for s in retrieved_raw
        ]

        correct = [
            MinimalSource(**s) if isinstance(s, dict) else s
            for s in correct_raw
        ]

        for k in ks:
            scores[k].append(
                recall_at_k(retrieved, correct, k)
            )

    return {
        f"recall@{k}": (
            sum(vals) / len(vals) if vals else 0.0
        )
        for k, vals in scores.items()
    }
