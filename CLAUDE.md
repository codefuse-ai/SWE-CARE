# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SWE-CARE is a comprehensive benchmark for evaluating Large Language Models (LLMs) on software engineering tasks, with a focus on code analysis, review, and issue-resolving capabilities. The project currently supports Python and Java.

The benchmark features two main task types:

1. **Issue Resolving**: Generate code patches to fix GitHub issues
2. **Code Review**: Generate comprehensive code review reports for code diffs

## Commands

### Development Setup

```bash
# Install dependencies using uv (recommended)
pip install uv
uv sync

# Or using pip
pip install -e .

# Install pre-commit hooks (for development)
pre-commit install
```

### Linting

```bash
# Run ruff linter (configured in pyproject.toml)
ruff check .

# Run ruff formatter
ruff format .

# Pre-commit runs both automatically
pre-commit run --all-files
```

### Running Tests

Note: This project doesn't have traditional unit tests. Instead, it focuses on data collection, inference, and evaluation scripts.

## High-Level Architecture

### Core Modules

1. **`src/swe_care/collect/`** - Data collection pipeline
   - `get_top_repos.py` - Find most starred repos by language
   - `get_graphql_prs_data.py` - Fetch PR data via GitHub GraphQL API
   - `classify_prs_data.py` - Analyze commits and label review comments
   - `build_code_review_dataset.py` - Build final dataset with LLM-classified metadata
   - `convert_to_rm_samples.py` - Convert to reward model training samples

2. **`src/swe_care/inference/`** - LLM inference pipeline
   - `create_code_review_text.py` - Generate text datasets with different context strategies
   - `run_api.py` - Run LLM inference on code review tasks

3. **`src/swe_care/harness/`** - Evaluation framework
   - `code_review_eval.py` - Evaluate model predictions using rule-based or LLM-based evaluators

4. **`src/swe_care/schema/`** - Data models
   - `dataset.py` - Core task instance schemas (IssueResolvingTaskInstance, CodeReviewTaskInstance)
   - `collect.py` - GitHub PR data schemas
   - `inference.py` - Inference input/output schemas
   - `evaluation.py` - Evaluation result schemas

5. **`src/swe_care/utils/`** - Utility functions
   - `github.py` - GitHub API interactions
   - `llm_models/clients.py` - LLM API clients (OpenAI, Anthropic, etc.)
   - `bm25_retrieval.py` - BM25-based file retrieval
   - `patch.py` - Patch file manipulation

### Key Patterns

- **Modular CLI**: Each module (`collect`, `inference`, `harness`) has its own `__main__.py` with subcommands
- **Schema-driven**: All data structures use dataclasses with JSON serialization
- **Parallel Processing**: Most operations support `--jobs` for concurrent execution
- **GitHub API Token Management**: Supports multiple tokens for rate limit handling

### Data Flow

1. **Collection**: GitHub repos → PR data → Classified PRs → Code review dataset
2. **Inference**: Dataset → Text generation → LLM predictions
3. **Evaluation**: Predictions + Dataset → Evaluation results

## Important Considerations

- **GitHub API Rate Limits**: Always provide GitHub tokens via `--tokens` parameter
- **LLM API Keys**: Set environment variables (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
- **Large Files**: Be careful with retrieval operations on large repositories
- **Parallel Jobs**: Adjust `--jobs` based on API rate limits and system resources

## Environment Variables

- `OPENAI_API_KEY` - OpenAI API key for GPT models
- `ANTHROPIC_API_KEY` - Anthropic API key for Claude models
- `OPENAI_BASE_URL` - Custom OpenAI-compatible API endpoint
- `ANTHROPIC_BASE_URL` - Custom Anthropic-compatible API endpoint
