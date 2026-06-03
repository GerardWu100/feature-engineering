"""Packaged project-level entrypoint for the simple feature pipeline."""

from __future__ import annotations

from run import main as run_main


def main() -> None:
    """Delegate to the simple feature-pipeline CLI."""
    run_main()


if __name__ == "__main__":
    main()
