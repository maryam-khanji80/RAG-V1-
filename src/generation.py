from typing import Any, Callable, List, Optional

from .models import MinimalSource
from .retrieval import Retriever

MAX_CONTEXT_CHARS = 6000


class AnswerGenerator:
    """Generates natural language answers from retrieved context."""

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        """Initialize the generator without loading the model.

        Args:
            model_name: HuggingFace model identifier to load on demand.
        """
        self.model_name = model_name
        self._pipeline: Optional[Callable[..., Any]] = None

    def _load(self) -> None:
        """Load the transformers text-generation pipeline (once)."""
        if self._pipeline is not None:
            return

        print(f"Loading model '{self.model_name}'...")

        from transformers import pipeline

        self._pipeline = pipeline(
            "text-generation",
            model=self.model_name,
            trust_remote_code=True,
        )

        print("Model loaded.")

    def generate(
        self,
        question: str,
        sources: List[MinimalSource],
        retriever: Retriever,
    ) -> str:
        """Generate an answer to a question given retrieved sources.

        Args:
            question: The question to answer.
            sources: Retrieved MinimalSource objects providing context.
            retriever: A Retriever used to read chunk text from disk.

        Returns:
            A generated, self-contained answer string.
        """
        self._load()

        if self._pipeline is None:
            return "Generation failed: model pipeline could not be loaded."

        context_parts: List[str] = []
        total_len = 0

        for source in sources:
            text = retriever.get_chunk_text(source)

            if not text:
                continue

            snippet = f"[Source: {source.file_path}]\n{text}"
            remaining = MAX_CONTEXT_CHARS - total_len

            if remaining <= 0:
                break

            if len(snippet) > remaining:
                snippet = snippet[:remaining]

            context_parts.append(snippet)
            total_len += len(snippet)

        context = "\n\n---\n\n".join(context_parts)

        prompt = (
            "You are a helpful assistant answering questions about the "
            "vLLM codebase. Use ONLY the context provided below. "
            "Be concise, self-contained, and cite the relevant file(s).\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        try:
            messages = [{"role": "user", "content": prompt}]

            outputs = self._pipeline(
                messages,
                max_new_tokens=512,
                do_sample=False,
            )

            generated = outputs[0]["generated_text"]

            if isinstance(generated, list):
                answer: str = generated[-1]["content"]
            else:
                answer = str(generated)

            return answer.strip()

        except Exception as exc:  # noqa: BLE001
            return f"Generation failed: {exc}"
