.PHONY: install run debug clean lint lint-strict moulinette pipeline

install:
	uv sync

run:
	uv run python -m src $(ARGS)

debug:
	uv run python -m pdb -m src $(ARGS)

clean:
	rm -rf __pycache__ .mypy_cache src/src/__pycache__ \
		data/processed data/output .pytest_cache

lint:
	uv run flake8 src
	uv run mypy src --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
lint-strict:
	uv run flake8 .
	uv run mypy . --strict

index:
	uv run python3 -m src index $(L)

search_dataset:
	uv run python -m src search_dataset --dataset_path $(QUESTIONS) --k $(K) --save_directory $(SAVE)

moulinette:
	./moulinette evaluate_student_search_results $(SEARCH_RESULT) $(ANSWERS) --k $(K) --max_context_length $(L)