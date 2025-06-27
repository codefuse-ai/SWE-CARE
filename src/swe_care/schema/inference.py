from dataclasses import dataclass

from dataclasses_json import dataclass_json

from swe_care.schema.dataset import CodeReviewTaskInstance


@dataclass_json
@dataclass
class CodeReviewInferenceInstance(CodeReviewTaskInstance):
    """Schema for code review inference instances."""

    text: str
    """The input text including instructions, the "Oracle"/RAG retrieved file, and an example of the patch format for output."""
