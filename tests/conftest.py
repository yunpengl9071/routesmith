# tests/conftest.py
import os
import pathlib
import sys

# RS_TEST_MOCK_LITELLM shim: stub litellm before routesmith imports it
if os.environ.get("RS_TEST_MOCK_LITELLM") == "1":
    import sys as _sys
    from unittest.mock import MagicMock
    _sys.modules.setdefault("litellm", MagicMock())

# Add src/ to path so tests can import from routesmith
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
