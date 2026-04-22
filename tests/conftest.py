"""pytest configuration and fixtures for universe tests"""

import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory"""
    return Path(__file__).parent.parent


@pytest.fixture(scope="function")
def temp_cache_dir(tmp_path: Path) -> Path:
    """Provide a temporary cache directory for tests"""
    cache_dir = tmp_path / "llm_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
