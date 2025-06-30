from pathlib import Path

from loguru import logger

from swe_care.schema.dataset import CodeReviewTaskInstance
from swe_care.schema.evaluation import CodeReviewPrediction
from swe_care.schema.inference import CodeReviewInferenceInstance


def load_code_review_dataset(dataset_file: Path | str) -> list[CodeReviewTaskInstance]:
    """Load the code review dataset instances from the JSONL file."""
    if isinstance(dataset_file, str):
        dataset_file = Path(dataset_file)
    logger.info("Loading dataset instances...")

    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    dataset_instances: list[CodeReviewTaskInstance] = []

    with open(dataset_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                instance = CodeReviewTaskInstance.from_json(line.strip())
                dataset_instances.append(instance)
            except Exception as e:
                logger.error(f"Error processing line {line_num}: {e}")
                raise e

    logger.success(f"Loaded {len(dataset_instances)} dataset instances")
    return dataset_instances


def load_code_review_predictions(
    predictions_path: Path | str,
) -> list[CodeReviewPrediction]:
    """Load the code review predictions from the JSONL file."""
    if isinstance(predictions_path, str):
        predictions_path = Path(predictions_path)
    logger.info("Loading predictions...")

    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    predictions: list[CodeReviewPrediction] = []

    with open(predictions_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                prediction = CodeReviewPrediction.from_json(line.strip())
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Error processing line {line_num}: {e}")
                raise e
    logger.success(f"Loaded {len(predictions)} predictions")
    return predictions


def load_code_review_text(
    dataset_file: Path | str,
) -> list[CodeReviewInferenceInstance]:
    """Load the code review text dataset instances from the JSONL file.

    Args:
        dataset_file: Path to the input JSONL file containing CodeReviewInferenceInstance objects

    Returns:
        List of CodeReviewInferenceInstance objects

    Raises:
        FileNotFoundError: If the dataset file doesn't exist
        Exception: If there's an error parsing the file
    """
    if isinstance(dataset_file, str):
        dataset_file = Path(dataset_file)
    logger.info("Loading inference text dataset instances...")

    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    dataset_instances: list[CodeReviewInferenceInstance] = []

    with open(dataset_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                instance = CodeReviewInferenceInstance.from_json(line.strip())
                dataset_instances.append(instance)
            except Exception as e:
                logger.error(f"Error processing line {line_num}: {e}")
                raise e

    logger.success(f"Loaded {len(dataset_instances)} inference text dataset instances")
    return dataset_instances
