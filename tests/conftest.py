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
    """Raised when code under test tries to reach another host."""


_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def _check(address: object) -> None:
    """Allow the loopback interface only.

    Event loops connect a socket pair over loopback to wake themselves up (on Windows through
    ``socket.socket``), and FastAPI's test client runs one. Refusing every socket made the API
    tests hang on Windows; refusing every host but this machine keeps the suite offline.
    """
    host = address[0] if isinstance(address, tuple) else address
    if isinstance(host, bytes):
        host = host.decode(errors="replace")
    if isinstance(host, str) and (host in _LOOPBACK or host.startswith("/")):
        return
    raise NetworkAccessError(f"tests must run offline: a connection to {host!r} was requested")


class _OfflineSocket(socket.socket):
    def connect(self, address):  # type: ignore[override]
        _check(address)
        return super().connect(address)

    def connect_ex(self, address):  # type: ignore[override]
        _check(address)
        return super().connect_ex(address)

    def sendto(self, data, *args):  # type: ignore[override]
        _check(args[-1])
        return super().sendto(data, *args)


def _create_connection(address, *args, **kwargs):
    _check(address)
    return _original_create_connection(address, *args, **kwargs)


_original_create_connection = socket.create_connection


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "socket", _OfflineSocket)
    monkeypatch.setattr(socket, "create_connection", _create_connection)
