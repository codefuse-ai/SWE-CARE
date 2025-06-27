# SWE-CARE: Software Engineering - Code Analysis and Review Evaluation

A comprehensive benchmark for evaluating Large Language Models (LLMs) on software engineering tasks, with a focus on code analysis, review, and issue-resolving capabilities.

## 📝 Overview

The primary goal of SWE-CARE is to assess LLMs in the following areas:

* **Solving Complex Programming Problems**: Evaluating the model's capability to understand, locate, and fix issues in real codebases.
* **Code Change Analysis**: Assessing the model's ability to analyze code changes, identify potential problems, and suggest improvements.
* **Complex Code Reasoning**: Measuring the model's deep analysis and reasoning skills regarding code logic, structure, and functionality.
* **Code Review Generation**: Evaluating the model's understanding of the logic behind a fix by generating a human-readable code review report.

SWE-CARE features two main task types:

1. **Issue Resolving**: Given a problem description (e.g., a GitHub issue), the model must generate a code patch to fix it. Evaluation is done by applying the patch and running tests in a reproducible environment.
2. **Code Review**: Given a code diff, the model must generate a comprehensive code review report. The quality of the report is assessed using a combination of automated metrics and LLM-as-a-judge evaluation.

The benchmark currently supports Python and Java.

## 🛠️ Set Up

Follow these steps to set up the project locally.

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/SWE-CARE.git
    cd SWE-CARE
    ```

2. **Install dependencies:**
    This project uses `uv` for package management. Make sure you have Python 3.10 or higher.

    ```bash
    pip install uv
    uv sync
    ```

    Alternatively, you can use `pip`:

    ```bash
    pip install -e .
    ```

3. **Set up pre-commit hooks (for development):**
    This project uses `ruff` for linting and formatting. The pre-commit hooks will run these checks automatically before each commit.

    ```bash
    pre-commit install
    ```

## 📊 Data Collection

The data collection process involves several steps to gather and process data from GitHub. The main scripts for this process are located in `src/swe_care/collect`.

Here's an example of the command-line usage for each step:

1. **Get Top Repositories**: Find the most starred repositories for a given language.

    ```bash
    python -m swe_care.collect get_top_repos \
        --language "Python" \
        --top-n 100 \
        --output-dir "results/top_repos" \
        --tokens "your_github_pat"
    ```

2. **Get Pull Request Data**: Fetch PR data from a specific repository using the GitHub GraphQL API.

    ```bash
    python -m swe_care.collect get_graphql_prs_data \
        --repo "<repo_owner>/<repo_name>" \
        --output-dir "results/graphql_prs_data" \
        --tokens "your_github_pat" \
        --max-number 20
    ```

3. **Evaluate Commits**: Evaluate the collected commits from the PRs.

    ```bash
    python -m swe_care.collect evaluate_commits \
        --graphql-prs-data-file "results/graphql_prs_data/<repo_owner>__<repo_name>_graphql_prs_data.jsonl" \
        --output-dir "./results/evaluate_commits"
    ```

4. **Build Code Review Dataset**: Build the final dataset for the code review task.

    ```bash
    python -m swe_care.collect build_code_review_dataset \
        --graphql-prs-data-file "results/graphql_prs_data/<repo_owner>__<repo_name>_graphql_prs_data.jsonl" \
        --pr-commits-evaluation-file "results/evaluate_commits/<repo_owner>__<repo_name>_pr_commits_evaluation.jsonl" \
        --output-dir "./results/dataset" \
        --tokens "your_github_pat"
    ```

You can find more details about the arguments for each script by running `python -m swe_care.collect -h`.

## 🔄 Inference

Before running evaluation, you can generate text datasets from the collected SWE-CARE data with different context strategies. The inference module creates datasets in the format required for LLM evaluation.

Here's an example of how to generate text datasets:

```bash
python -m swe_care.inference create_code_review_text \
    --dataset-file "results/dataset/code_review_task_instances.jsonl" \
    --output-dir "results/code_review_text" \
    --file-source "oracle" \
    --tokens "your_github_pat"
```

### File Source Strategies

The `--file-source` parameter supports different strategies for selecting context files:

* **oracle**: Uses ground truth files (files that were actually changed in both the review commit and merged commit)
* **bm25**: Uses BM25 retrieval to select relevant files (requires `--retrieval-file`)
* **all**: Uses all available files up to a specified limit (requires `--k` parameter)

### Additional Parameters

* `--k`: Maximum number of files to include (used with bm25 and all strategies)
* `--retrieval-file`: Path to BM25 retrieval results file (required for bm25 strategy)
* `--tokens`: GitHub Personal Access Token(s) for API access

The generated text dataset will contain prompts with code context, issue descriptions, patches, and review instructions formatted for LLM evaluation.

## 🚀 Evaluation

The evaluation harness is used to assess model predictions on the code review task. The main script is `src/swe_care/harness/code_review_eval.py`.

Here's an example of how to run the evaluation:

```bash
export OPENAI_API_KEY=<your_openai_api_key>
python -m swe_care.harness code_review_eval \
    --dataset-file "results/code_review_task_instances.jsonl" \
    --predictions-path "results/code_review_predictions.jsonl" \
    --output-dir "./results/report" \
    --evaluator "llm_evaluator" \
    --llm-model "Your-LLM-Model" \
    --llm-base-url "https://your.llm.provider/v1"
```

The evaluator will compare the model's generated reviews against the reference reviews in the dataset and produce evaluation metrics. The supported evaluators are defined in `src/swe_care/harness/evaluators`.

## 📜 Citation

(To be added)

## 🙏 Acknowledgements

(To be added)
