from pathlib import Path
from typing import Optional


def build_code_review_dataset(
    graphql_prs_data_file: Path,
    output_dir: Path = None,
    tokens: Optional[list[str]] = None,
) -> None:
    """
    Build code review task dataset.

    Args:
        graphql_prs_data_file: Path to GraphQL PRs data file (output from get_graphql_prs_data)
        output_dir: Directory to save the output data
        tokens: Optional list of GitHub tokens for API requests
    """
    # TODO
    pass
