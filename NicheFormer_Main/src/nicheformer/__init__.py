try:
    from importlib.metadata import version
    __version__ = version("nicheformer")
except Exception:
    __version__ = "0.0.1"

from . import data, models

__all__ = ["data", "models"]
