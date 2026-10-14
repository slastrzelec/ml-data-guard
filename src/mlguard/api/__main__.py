"""Run the service: ``python -m mlguard.api`` (HOST and PORT, default 0.0.0.0:7860).

Uvicorn's own access log is switched off because it records client addresses and query strings;
the service logs method, path, status and duration instead (SPEC section 17.2).
"""

from __future__ import annotations

import logging
import os

import uvicorn


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    uvicorn.run(
        "mlguard.api:app",
        host=os.environ.get("HOST", "0.0.0.0"),  # noqa: S104 - inside a container
        port=int(os.environ.get("PORT", "7860")),
        access_log=False,
        proxy_headers=False,
        server_header=False,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
