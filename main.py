"""Command-line interface for the Blog Generator.

This module provides the CLI entry point for running the blog
generation pipeline from the terminal.
"""

import argparse
import logging
import sys
from typing import NoReturn

from config import settings
from core.exceptions import BlogGenerationError, ConfigurationError
from core.orchestrator import Orchestrator
from utils.logger import get_logger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the blog generation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=settings.default_topic,
        help=f"Blog topic to generate (default: {settings.default_topic})",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    return parser.parse_args()


def main() -> int:
    """Run the blog generation pipeline.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    logger = get_logger(__name__)

    try:
        args = parse_args()

        # Adjust log level if verbose
        if args.verbose:
            logger.setLevel(logging.DEBUG)
            logging.getLogger().setLevel(logging.DEBUG)

        logger.info(f"Starting blog generation for topic: {args.topic}")

        orchestrator = Orchestrator()
        result = orchestrator.run_sync(topic=args.topic)

        print("\n" + "=" * 60)
        print("GENERATED BLOG POST")
        print("=" * 60 + "\n")
        print(result)
        print("\n" + "=" * 60)

        logger.info("Blog generation completed successfully")
        return 0

    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        print(f"\nError: {e}", file=sys.stderr)
        print("\nPlease ensure you have configured your .env file", file=sys.stderr)
        print("Copy .env.template to .env and add your OPENROUTER_API_KEY", file=sys.stderr)
        return 1

    except BlogGenerationError as e:
        logger.error(f"Generation error: {e}")
        print(f"\nError generating blog: {e}", file=sys.stderr)
        return 1

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        print("\nOperation cancelled.", file=sys.stderr)
        return 130

    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())