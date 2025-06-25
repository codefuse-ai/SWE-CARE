from pathlib import Path

# --- GraphQL Queries Directory ---
GRAPHQL_QUERIES_DIR = Path(__file__).parent

# --- GraphQL Queries Mapping ---
GRAPHQL_QUERIES = {
    # Main query for fetching PRs with first page of nested data
    "merged_pull_requests": (
        GRAPHQL_QUERIES_DIR / "GetMergedPullRequests.graphql"
    ).read_text(),
    # Query for fetching additional pages of labels
    "labels": (GRAPHQL_QUERIES_DIR / "GetLabels.graphql").read_text(),
    # Query for fetching additional pages of commits
    "commits": (GRAPHQL_QUERIES_DIR / "GetCommits.graphql").read_text(),
    # Query for fetching additional pages of reviews
    "reviews": (GRAPHQL_QUERIES_DIR / "GetReviews.graphql").read_text(),
    # Query for fetching additional pages of review comments
    "review_comments": (GRAPHQL_QUERIES_DIR / "GetReviewComments.graphql").read_text(),
    # Query for fetching additional pages of closing issues
    "closing_issues": (GRAPHQL_QUERIES_DIR / "GetClosingIssues.graphql").read_text(),
    # Query for fetching additional pages of issue labels
    "issue_labels": (GRAPHQL_QUERIES_DIR / "GetIssueLabels.graphql").read_text(),
    # Query for fetching additional pages of issue comments
    "issue_comments": (GRAPHQL_QUERIES_DIR / "GetIssueComments.graphql").read_text(),
    # Query for fetching additional pages of review threads
    "review_threads": (GRAPHQL_QUERIES_DIR / "GetReviewThreads.graphql").read_text(),
    # Query for fetching additional pages of thread comments
    "thread_comments": (GRAPHQL_QUERIES_DIR / "GetThreadComments.graphql").read_text(),
    # Query for fetching additional pages of specific PR
    "specific_pr": (GRAPHQL_QUERIES_DIR / "GetSpecificPullRequest.graphql").read_text(),
}
