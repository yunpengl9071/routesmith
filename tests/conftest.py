# tests/conftest.py
import sys
import pathlib

# Add src/ to path so tests can import from routesmith
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
