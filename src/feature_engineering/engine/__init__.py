"""Feature engines for cached batch transforms and live updates.

- ``FeatureEngine`` (``batch.py``): validate/resolve the config once, then
  ``transform(frame) -> frame`` for research and backtesting.
- ``OnlineFeatureEngine`` (``online.py``): constant-time incremental updates for
  live trading via ``update(bar) -> {feature: value}``.
"""

from feature_engineering.engine.batch import FeatureEngine
from feature_engineering.engine.online import OnlineFeatureEngine

__all__ = ["FeatureEngine", "OnlineFeatureEngine"]
