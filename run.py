"""Root wrapper for the simple feature pipeline CLI.

Keeping this file lets local workflows continue to use:

    uv run python run.py --config config.toml
"""

from __future__ import annotations

from feature_engineering.pipeline.cli import main


if __name__ == "__main__":
    main()
