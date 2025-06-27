import argparse
import sys
from pathlib import Path

from loguru import logger

import swe_care.inference.create_code_review_text
from swe_care.inference.create_code_review_text import create_code_review_text

# Mapping of subcommands to their function names
SUBCOMMAND_MAP = {
    "create_code_review_text": {
        "function": create_code_review_text,
        "help": swe_care.inference.create_code_review_text.__doc__,
    },
}


def create_global_parser():
    """Create a parser with global arguments that can be used as a parent parser."""
    global_parser = argparse.ArgumentParser(add_help=False)
    global_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Path to output directory",
    )
    return global_parser


def get_args():
    # Parse command line manually to handle flexible argument order
    args = sys.argv[1:]

    # Find the subcommand
    subcommands = list(SUBCOMMAND_MAP.keys())
    subcommand = None
    subcommand_index = None

    for i, arg in enumerate(args):
        if arg in subcommands:
            subcommand = arg
            subcommand_index = i
            break

    # Create global parser
    global_parser = create_global_parser()

    if subcommand is None:
        # No subcommand found, use normal argparse
        parser = argparse.ArgumentParser(
            prog="swe_care.inference",
            description="Inference tools for SWE-CARE",
            parents=[global_parser],
        )

        subparsers = parser.add_subparsers(dest="command", help="Available commands")
        for cmd, info in SUBCOMMAND_MAP.items():
            subparsers.add_parser(cmd, help=info["help"])

        return parser.parse_args(args)

    # Create the appropriate subcommand parser with global parser as parent
    match subcommand:
        case "create_code_review_text":
            sub_parser = argparse.ArgumentParser(
                prog=f"swe_care.inference {subcommand}",
                parents=[global_parser],
                description=SUBCOMMAND_MAP[subcommand]["help"],
            )
            sub_parser.add_argument(
                "--dataset-file",
                type=Path,
                required=True,
                help="Path to the input SWE-CARE dataset file",
            )
            sub_parser.add_argument(
                "--file-source",
                type=str,
                choices=["oracle", "bm25", "all"],
                required=True,
                help="Source strategy for files: 'oracle' (ground truth), 'bm25' (retrieval), or 'all' (all available)",
            )
            sub_parser.add_argument(
                "--k",
                type=int,
                required=False,
                default=None,
                help="Maximum number of files to use for retrieval",
            )
            sub_parser.add_argument(
                "--retrieval-file",
                type=Path,
                default=None,
                help="File with BM25 retrieval results (required when --file-source is 'bm25')",
            )
            sub_parser.add_argument(
                "--tokens",
                type=str,
                nargs="*",
                default=None,
                help="GitHub API token(s) to be used randomly for fetching data",
            )

    # Parse all arguments with the subcommand parser
    # This will include both global and subcommand-specific arguments
    # Remove the subcommand itself from args
    args_without_subcommand = args[:subcommand_index] + args[subcommand_index + 1 :]
    final_namespace = sub_parser.parse_args(args_without_subcommand)
    final_namespace.command = subcommand

    return final_namespace


def main():
    args = get_args()

    if args.command in SUBCOMMAND_MAP:
        # Get the function from the mapping
        cmd_info = SUBCOMMAND_MAP[args.command]
        function = cmd_info["function"]

        # Prepare common arguments
        common_kwargs = {"output_dir": args.output_dir}

        # Add specific arguments based on subcommand
        match args.command:
            case "create_code_review_text":
                function(
                    dataset_file=args.dataset_file,
                    file_source=args.file_source,
                    k=args.k,
                    retrieval_file=args.retrieval_file,
                    tokens=args.tokens,
                    **common_kwargs,
                )
    else:
        logger.info("Please specify a command. Use --help for available commands.")


if __name__ == "__main__":
    main()
