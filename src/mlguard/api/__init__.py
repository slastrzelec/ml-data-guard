"""HTTP service and browser front end (SPEC section 17). Needs the ``api`` extra.

pip install "ml-data-guard[api]"
uvicorn mlguard.api:app --no-access-log
"""

from mlguard.api.service import app, create_app

__all__ = ["app", "create_app"]
