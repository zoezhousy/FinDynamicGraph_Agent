"""Deterministic portfolio manager — re-exported from roles.py.

This module exists as the canonical import path for the deterministic PM.
The actual implementation lives in ``roles.py`` for backward compatibility
with existing tests and imports.
"""

from src.agents.roles import portfolio_manager_decide

__all__ = ["portfolio_manager_decide"]
