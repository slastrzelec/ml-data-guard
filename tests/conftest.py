"""Shared test configuration: no network, deterministic property-based tests."""

from __future__ import annotations

import os
import socket

import pytest
from hypothesis import settings

# Deterministic by default (same examples on every run and in CI).
# HYPOTHESIS_PROFILE=explore searches with fresh random examples.
settings.register_profile("deterministic", derandomize=True, deadline=None, max_examples=200)
settings.register_profile("explore", deadline=None, max_examples=2000)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "deterministic"))


class NetworkAccessError(RuntimeError):
    """Raised when code under test tries to open a socket."""


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: object, **kwargs: object) -> None:
        raise NetworkAccessError("tests must run offline: a socket was requested")

    monkeypatch.setattr(socket, "socket", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
