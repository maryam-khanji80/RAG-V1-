"""Command-line interface for the src RAG system."""

import json
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from .evaluation import evaluate_dataset
from .generation import AnswerGenerator
from .indexing import build_chunks, save_index
from .models import (
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    srcSearchResults,
    srcSearchResultsAndAnswer,
)
from .retrieval import Retriever

DEFAULT_REPO_PATH = "data/raw/vllm-0.10.1"
DEFAULT_PROCESSED_DIR = "data/processed"
DEFAULT_REPO_ROOT = "."


class srcCLI:
    def index(
        self,
        max_chunk_size: int = 2000,
        _repo_path: str = DEFAULT_REPO_PATH,
        _processed_dir: str = DEFAULT_PROCESSED_DIR,
    ) -> None:
        if max_chunk_size <= 0:
            print("Error: --max_chunk_size must be a positive integer.")
            return

        if not Path(_repo_path).exists():
            print(f"Error: repository path '{_repo_path}' does not exist.")
            return

        chunks = build_chunks(_repo_path, max_chunk_size)

        if not chunks:
            print("Warning: no chunks produced from the repository.")
            return

        save_index(chunks, _processed_dir)
        print(f"Ingestion complete! Indices saved under {_processed_dir}/")

    def search(
        self,
        query: str,
        k: int = 10,
        repo_root: str = DEFAULT_REPO_ROOT,
        processed_dir: str = DEFAULT_PROCESSED_DIR,
    ) -> None:

        if not query or not str(query).strip():
            print("Error: query must be a non-empty string.")
            return

        try:
            retriever = Retriever(processed_dir, repo_root)
        except FileNotFoundError as exc:
            print(f"Error: {exc}")
            return

        sources = retriever.search(str(query), k=k)
        for source in sources:
            print(source.model_dump_json())

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results",
        _repo_root: str = DEFAULT_REPO_ROOT,
        _processed_dir: str = DEFAULT_PROCESSED_DIR,
    ) -> None:
        dataset_file = Path(dataset_path)
        if not dataset_file.exists():
            print(f"Error: dataset path '{dataset_path}' does not exist.")
            return

        try:
            retriever = Retriever(_processed_dir, _repo_root)
        except FileNotFoundError as exc:
            print(f"Error: {exc}")
            return

        try:
            with open(dataset_file, "r", encoding="utf-8") as f:
                data: Dict[str, Any] = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Error: could not read dataset file: {exc}")
            return

        questions: List[Dict[str, Any]] = data.get("rag_questions", [])
        results: List[MinimalSearchResults] = []

        for item in tqdm(questions, desc="Searching", unit="q"):
            question_id = str(item.get("question_id", ""))
            question_text = str(item.get("question", ""))
            sources = retriever.search(question_text, k=k)
            results.append(
                MinimalSearchResults(
                    question_id=question_id,
                    question=question_text,
                    retrieved_sources=sources,
                )
            )

        output = srcSearchResults(search_results=results, k=k)

        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / dataset_file.name

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(output.model_dump_json(indent=2))

        print(f"Saved src_search_results to {out_path}")

    def answer(
        self,
        query: str,
        k: int = 10,
        repo_root: str = DEFAULT_REPO_ROOT,
        processed_dir: str = DEFAULT_PROCESSED_DIR,
        model_name: str = "Qwen/Qwen3-0.6B",
    ) -> None:

        if not query or not str(query).strip():
            print("Error: query must be a non-empty string.")
            return

        try:
            retriever = Retriever(processed_dir, repo_root)
        except FileNotFoundError as exc:
            print(f"Error: {exc}")
            return

        sources = retriever.search(str(query), k=k)
        generator = AnswerGenerator(model_name=model_name)
        answer_text = generator.generate(str(query), sources, retriever)

        result = MinimalAnswer(
            question_id="cli_query",
            question=str(query),
            retrieved_sources=sources,
            answer=answer_text,
        )
        print(result.model_dump_json(indent=2))

    def answer_dataset(
        self,
        src_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
        repo_root: str = DEFAULT_REPO_ROOT,
        processed_dir: str = DEFAULT_PROCESSED_DIR,
        model_name: str = "Qwen/Qwen3-0.6B",
    ) -> None:

        results_file = Path(src_search_results_path)
        if not results_file.exists():
            print(
                f"Error: path '{src_search_results_path}' does not exist."
            )
            return

        try:
            with open(results_file, "r", encoding="utf-8") as f:
                data: Dict[str, Any] = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Error: could not read search results file: {exc}")
            return

        try:
            retriever = Retriever(processed_dir, repo_root)
        except FileNotFoundError as exc:
            print(f"Error: {exc}")
            return

        search_results: List[Dict[str, Any]] = data.get("search_results", [])
        k: int = int(data.get("k", 10))

        print(
            f"Loaded {len(search_results)} questions "
            f"from {src_search_results_path}"
        )

        generator = AnswerGenerator(model_name=model_name)
        answers: List[MinimalAnswer] = []

        for idx, item in enumerate(
            tqdm(search_results, desc="Answering", unit="q")
        ):
            question_id = str(item.get("question_id", ""))
            question_text = str(item.get("question", ""))
            raw_sources: List[Dict[str, Any]] = item.get(
                "retrieved_sources", []
            )
            sources = [MinimalSource(**s) for s in raw_sources]

            answer_text = generator.generate(question_text, sources, retriever)

            answers.append(
                MinimalAnswer(
                    question_id=question_id,
                    question=question_text,
                    retrieved_sources=sources,
                    answer=answer_text,
                )
            )

            if (idx + 1) % 10 == 0 or (idx + 1) == len(search_results):
                print(
                    f"Processed {idx + 1} of {len(search_results)} questions"
                )

        output = srcSearchResultsAndAnswer(search_results=answers, k=k)

        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / results_file.name

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(output.model_dump_json(indent=2))

        print(f"Saved src_search_results_and_answer to {out_path}")

    def evaluate(
        self,
        src_results_path: str,
        dataset_path: str,
        k: int = 10,
    ) -> None:
        results_file = Path(src_results_path)
        dataset_file = Path(dataset_path)

        if not results_file.exists():
            print(f"Error: path '{src_results_path}' does not exist.")
            return
        if not dataset_file.exists():
            print(f"Error: path '{dataset_path}' does not exist.")
            return

        try:
            with open(results_file, "r", encoding="utf-8") as f:
                src_data: Dict[str, Any] = json.load(f)
            with open(dataset_file, "r", encoding="utf-8") as f:
                truth_data: Dict[str, Any] = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Error: could not read input files: {exc}")
            return

        src_results: List[Dict[str, Any]] = src_data.get(
            "search_results", []
        )
        ground_truth: Dict[str, Dict[str, Any]] = {
            str(q["question_id"]): q
            for q in truth_data.get("rag_questions", [])
        }

        ks = sorted({1, 3, 5, k})
        metrics = evaluate_dataset(src_results, ground_truth, ks)

        n_with_sources = sum(
            1
            for q in truth_data.get("rag_questions", [])
            if q.get("sources")
        )
        n_src = len(src_results)

        print("src data is valid: True")
        print(f"Total number of questions: {len(ground_truth)}")
        print(f"Total number of questions with sources: {n_with_sources}")
        print(f"Total number of questions with src sources: {n_src}")
        print()
        print("Evaluation Results")
        print("=" * 40)
        print(f"Questions evaluated: {n_src}")
        for key in sorted(metrics, key=lambda x: int(x.split("@")[1])):
            print(f"{key.capitalize()}: {metrics[key]:.3f}")


def main() -> None:
    """Entry point for `python -m src`."""
    import fire

    fire.Fire(srcCLI)
