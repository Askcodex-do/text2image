"""AI Image Studio.

An image generation/editing application whose pipeline separates *identity*,
*content* and *style* so that a person's identity can be deliberately
preserved across a requested visual transformation.
"""

from .app import create_app

__version__ = "1.0.0"
__all__ = ["create_app", "__version__"]