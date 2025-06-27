import unidiff
from loguru import logger


def get_changed_file_paths(patch_content: str) -> list[str]:
    """
    Extract file paths that are changed in a patch.

    Args:
        patch_content: The patch content as a string

    Returns:
        A list of file paths that are modified in the patch
    """
    try:
        patch_set = unidiff.PatchSet(patch_content)
        changed_files = []

        for patched_file in patch_set:
            file_path = patched_file.source_file
            if file_path.startswith("a/"):
                file_path = file_path[2:]  # Remove 'a/' prefix

            if file_path == "/dev/null":
                logger.debug("Skipping /dev/null file, this is a new file")
                continue

            changed_files.append(file_path)

        return changed_files
    except Exception:
        # If parsing fails, return empty list
        return []
