*This project has been created as part of the 42 curriculum by mkhanji.*

# RAG Against the Machine

A Retrieval-Augmented Generation (RAG) system that answers questions about the vLLM codebase using BM25 retrieval and the Qwen3-0.6B language model.

## Description

This system ingests the vLLM repository, builds a searchable index, retrieves relevant code snippets and documentation for a given question, and generates grounded natural-language answers using a local LLM.

The system is evaluated using Recall@k metrics to measure retrieval quality.

The pipeline has four main stages:

1. **Ingest** — chunk Python files and Markdown documentation and build a BM25 index.
2. **Retrieve** — find the top-k most relevant chunks for a query.
3. **Generate** — pass the retrieved context to Qwen3-0.6B to produce an answer.
4. **Evaluate** — measure Recall@k against ground-truth source annotations.

## System Architecture

```text
vLLM repository
       │
       ▼
 chunking.py
       │
       │ AST-based chunking for Python
       │ paragraph-based chunking for Markdown
       ▼
 indexing.py
       │
       │ BM25 index using bm25s
       │ saved to data/processed/
       ▼
 retrieval.py
       │
       │ BM25 search and ranking
       │ returns MinimalSource objects
       ▼
 generation.py
       │
       │ Qwen3-0.6B via Transformers
       ▼
 evaluation.py
       │
       │ Recall@k evaluation
       ▼
 cli.py
       │
       ├── index
       ├── search
       ├── search_dataset
       ├── answer
       ├── answer_dataset
       └── evaluate
```

At query time, the CLI sends the user's question to the retriever. The retriever uses BM25 to rank indexed chunks and returns the top-k relevant sources. These sources are then passed as context to the language model, which generates the final answer.

## Chunking Strategy

Two chunking strategies are implemented depending on the file type.

### Python files (`.py`)

Python files use **AST-aware chunking**.

The file is parsed using Python's `ast` module and split at top-level `def` and `class` boundaries. This keeps functions and classes together instead of cutting them in the middle.

If a function or class is larger than the configured maximum chunk size, it is further split on newline boundaries.

### Markdown and text files

Markdown files use **paragraph-based chunking**.

Paragraphs separated by blank lines are greedily packed into chunks up to the configured maximum chunk size. Oversized paragraphs are split on newline boundaries.

The default maximum chunk size is **2000 characters** and can be configured with:

```bash
--max_chunk_size
```

For example:

```bash
uv run python -m src index --max_chunk_size 500
```

### Chunk Size Trade-off

Chunk size affects both retrieval quality and answer generation:

* **Smaller chunks** improve retrieval precision but can separate related information and lose context.
* **Larger chunks** preserve more context but may contain irrelevant information and increase LLM context usage.
* **2000 characters** provides a balance between context preservation and retrieval precision for this project.

## Retrieval Method

The project uses **BM25 (Best Match 25)** through the `bm25s` library.

BM25 is a probabilistic ranking algorithm based on term frequency and inverse document frequency. It also applies document-length normalization and term-frequency saturation.

For each query, the system:

1. Tokenizes the query.
2. Searches the BM25 index.
3. Calculates a relevance score for each candidate chunk.
4. Ranks the chunks by their BM25 scores.
5. Returns the top-k highest-ranked chunks.

BM25 was chosen because it is fast, requires no GPU, and performs well for exact keyword matching. This is particularly useful for a codebase where function names, class names, configuration options, and API terms are highly discriminative.

The index is saved to disk after ingestion and loaded when needed, allowing retrieval to remain fast after indexing.

## Performance Analysis

The following results come from the retrieval evaluation:

| Dataset | Recall@1 | Recall@3 | Recall@5  | Recall@10 |
| ------- | -------- | -------- | --------- | --------- |
| Docs    | 0.590    | 0.790    | **0.840** | 0.870     |
| Code    | 0.290    | 0.490    | **0.560** | 0.620     |

The required Recall@5 thresholds are:

* Documentation: **≥ 0.80**
* Code: **≥ 0.50**

The system achieved:

* **Docs Recall@5: 0.840 (84%)**
* **Code Recall@5: 0.560 (56%)**

Both retrieval requirements are therefore satisfied.

### System Performance

The retrieval evaluation also measured:

* **Indexing time:** 5 seconds
* **Warm retrieval:** 200 questions in approximately 1 second
* **Indexing requirement:** ≤ 300 seconds
* **Warm retrieval requirement:** ≤ 90 seconds for 200 questions

The system passed all four retrieval evaluation tests.

## Design Decisions

### BM25 over TF-IDF

BM25 was chosen instead of basic TF-IDF because it includes document-length normalization and term-frequency saturation.

This is useful for a codebase containing files and chunks of different sizes.

### Separate chunking strategies

Different strategies are used for Python and Markdown files.

AST-aware chunking preserves functions and classes as semantic units, while paragraph-based chunking is more appropriate for documentation.

### Lazy model loading

The Qwen3-0.6B model is loaded only when `answer` or `answer_dataset` is called.

This keeps indexing and retrieval commands fast and avoids loading the language model when it is not needed.

### Pydantic models

Pydantic models are used for the project's data structures and validation. This helps ensure that retrieved sources and generated outputs follow the expected schemas.

### `uv` package manager

`uv` is used for dependency management and reproducible environments through the project's lockfile.

## Trade-offs

The implementation involves several trade-offs:

* **Small chunks vs. large chunks:** Small chunks improve precision but can lose context. Large chunks preserve context but may introduce irrelevant information.
* **BM25 vs. semantic retrieval:** BM25 is fast and effective for exact code/API terminology, but it can miss semantically similar queries that use different words.
* **Local LLM vs. external API:** Running Qwen3-0.6B locally avoids external API dependencies, but CPU generation is slower.
* **Lazy loading vs. startup time:** Lazy loading keeps retrieval commands fast, but the first answer-generation request has model-loading overhead.
* **Top-k value:** A larger k provides more context to the LLM but can introduce irrelevant chunks and increase context size.

## Challenges Faced

### Disk space

The project dependencies, especially machine-learning packages and model dependencies, required significant disk space. The project environment was therefore managed carefully using `uv` and the available storage.

### File path consistency

The `file_path` stored in the retrieval index must match the paths used by the evaluation datasets.

This was solved by indexing the vLLM repository from the repository root so that relative paths remain consistent.

### Generation speed

Qwen3-0.6B runs locally on CPU and generation is significantly slower than retrieval.

The model is loaded once and reused for multiple questions in a batch instead of being loaded separately for every question.

## Moulinette Evaluation

### Retrieval evaluation

Run:

```bash
./exams/scripts/exam_retrieval.sh \
  --src-path . \
  --moulinette-path ./moulinette/moulinette-ubuntu \
  --module-name src
```

The retrieval evaluation checks:

* Indexing performance
* Warm retrieval throughput
* Documentation Recall@5
* Code Recall@5

### Answer evaluation

Run:

```bash
./exams/scripts/exam_answer.sh \
  --src-path . \
  --moulinette-path ./moulinette/moulinette-ubuntu \
  --module-name src
```

Example questions for answer evaluation include:

```text
What does the LLM.chat method in vLLM implement?

What is the CLI command to start vLLM serve?

What API endpoint does vLLM support for chat completions?
```

### Edge cases

The implementation also handles invalid or unusual inputs without producing Python tracebacks.

Examples include:

```bash
uv run python -m src search "" --k 10
```

```bash
uv run python -m src search "asdfghjkl" --k 10
```

```bash
uv run python -m src answer "What is vLLM?" --k 0
```

```bash
uv run python -m src search_dataset \
  --dataset_path /nonexistent.json
```

## Requirements

* Python 3.10+
* `uv` package manager
* Approximately 3 GB of disk space for dependencies, model weights, and indexes

## Installation

```bash
git clone <your-repository-url>

cd haeval

uv sync
```

## Setup Data

Place the vLLM repository at:

```text
data/raw/vllm-0.10.1/
```

Public datasets are stored under:

```text
data/datasets_public/public/AnsweredQuestions/
data/datasets_public/public/UnansweredQuestions/
```

For example:

```text
data/datasets_public/public/AnsweredQuestions/dataset_docs_public.json
data/datasets_public/public/AnsweredQuestions/dataset_code_public.json
data/datasets_public/public/UnansweredQuestions/dataset_docs_public.json
data/datasets_public/public/UnansweredQuestions/dataset_code_public.json
```

## Example Usage

### 1. Build the index

```bash
uv run python -m src index --max_chunk_size 2000
```

### 2. Search a single query

```bash
uv run python -m src search \
  "How does vLLM handle continuous batching?" \
  --k 10
```

### 3. Search a full dataset

```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets_public/public/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results
```

### 4. Evaluate retrieval quality

```bash
uv run python -m src evaluate \
  data/output/search_results/dataset_docs_public.json \
  data/datasets_public/public/AnsweredQuestions/dataset_docs_public.json \
  --k 10
```

### 5. Generate an answer

```bash
uv run python -m src answer \
  "What is PagedAttention?" \
  --k 10
```

### 6. Generate answers for a full dataset

```bash
uv run python -m src answer_dataset \
  --src_search_results_path \
  data/output/search_results/dataset_docs_public.json \
  --save_directory \
  data/output/search_results_and_answer
```

## RAG Pipeline Example

A typical question goes through the following pipeline:

```text
User Question
      │
      ▼
   Tokenize
      │
      ▼
 BM25 Retrieval
      │
      ▼
 Top-k Relevant Chunks
      │
      ▼
 Retrieved Context
      │
      ▼
 Qwen3-0.6B
      │
      ▼
 Generated Answer
```

For example:

```bash
uv run python -m src answer \
  "How to configure OpenAI server?" \
  --k 10
```

The retriever first finds relevant vLLM documentation and code. These retrieved chunks are then provided to Qwen3-0.6B as context so that the generated answer is grounded in the vLLM codebase.

## Resources

* [vLLM project](https://github.com/vllm-project/vllm)
* [BM25S library](https://github.com/xhluca/bm25s)
* [Qwen3-0.6B model on Hugging Face](https://huggingface.co/Qwen/Qwen3-0.6B)
* [Pydantic documentation](https://docs.pydantic.dev/)
* [BM25 original paper — Robertson & Zaragoza, 2009](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf)
* [RAG paper — Lewis et al., 2020](https://arxiv.org/abs/2005.11401)

## AI Usage

AI (Claude) was used for:

* Generating the initial boilerplate structure for source files
* Debugging import errors and attribute typos during development
* Writing docstrings and type hints

All generated code was reviewed, understood, and tested before use.

The chunking logic, retrieval pipeline, and evaluation metric were verified manually against the dataset ground truth.
