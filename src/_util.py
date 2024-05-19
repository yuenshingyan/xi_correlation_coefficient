from typing import Any


__all__ = ["_is_array_like"]
__author__ = "Yuen Shing Yan Hindy"
__version__ = "1.0.0"


def _is_array_like(x: Any) -> bool:
    """
    Checks if the given object is array-like.

    Parameters
    ----------
    x : Any
        The object to check.

    Returns
    -------
    bool
        True if x is array-like, False otherwise.
    """
    if hasattr(x, "__len__") and hasattr(x, '__getitem__'):
        return True

    return False
