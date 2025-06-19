from dataclasses import dataclass
from typing import Any

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class ResolvedIssue:
    """Schema for resolved issue instances."""

    number: int
    title: str
    body: str


@dataclass_json
@dataclass
class IssueResolvingTaskMetadata:
    """Schema for metadata instances."""

    problem_domains: list[str]
    difficulty: str


@dataclass_json
@dataclass
class IssueResolvingTaskInstance:
    """Schema for Issue Resolving task instances."""

    instance_id: str
    repo: str
    language: str
    pull_number: int
    title: str
    body: str
    created_at: str
    problem_statement: str
    hints_text: str
    resolved_issues: list[ResolvedIssue]
    base_commit: str
    patch: str
    test_patch: str
    env_setup_config: dict[str, Any]
    FAIL_TO_PASS: list[str]
    PASS_TO_PASS: list[str]
    version: str
    metadata: IssueResolvingTaskMetadata


@dataclass_json
@dataclass
class CodeReviewTaskMetadata:
    """Schema for code review metadata instances."""

    problem_domains: list[str]
    difficulty: str
    estimated_review_effort: int


@dataclass_json
@dataclass
class ReferenceReviewComment:
    """Schema for reference review comment instances."""

    body: str

    created_at: str
    updated_at: str
    path: str
    diff_hunk: str
    line: int
    start_line: int
    original_line: int
    original_start_line: int


@dataclass_json
@dataclass
class CommitToReview:
    """Schema for commit to review instances."""

    head_commit: str
    head_commit_message: str
    patch_to_review: str
    reference_review_comments: list[ReferenceReviewComment]


@dataclass_json
@dataclass
class CodeReviewTaskInstance:
    """Schema for Code Review task instances."""

    instance_id: str
    repo: str
    language: str
    pull_number: int
    title: str
    body: str
    created_at: str
    problem_statement: str
    hints_text: str
    resolved_issues: list[ResolvedIssue]
    base_commit: str
    commit_to_review: CommitToReview
    merge_commit: str
    merged_patch: str
    metadata: CodeReviewTaskMetadata
