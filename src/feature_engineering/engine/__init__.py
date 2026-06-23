"""Feature engines: cached batch transforms and live incremental updates.

- ``FeatureEngine`` (``batch.py``): validate/resolve the config once, then
  ``transform(df) -> df`` for research and backtesting.
- ``OnlineFeatureEngine`` (``online.py``): O(1)-per-bar incremental updates for
  live trading via ``update(bar) -> {feature: value}``.
"""

from feature_engineering.engine.batch import FeatureEngine
from feature_engineering.engine.online import (
    ONLINE_FEATURE_FACTORIES,
    OnlineFeatureEngine,
)

__all__ = ["FeatureEngine", "OnlineFeatureEngine", "ONLINE_FEATURE_FACTORIES"]
