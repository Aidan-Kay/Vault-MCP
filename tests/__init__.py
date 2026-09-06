"""Test harness.

src.config resolves settings at import time and refuses to start without a key,
so these defaults are set before anything under src is imported. The tests run
against the real vault read-only: the resolver's job is tolerating what this
vault actually contains, and a fixture would only prove it tolerates the fixture.
"""

import os

os.environ.setdefault("VAULT_MCP_API_KEY", "test")
os.environ.setdefault("VAULT_PATH", "/media/Share/Vault")
