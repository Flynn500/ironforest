import pytest
import numpy as np


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def tmp_path_str(tmp_path):
    """String version of tmp_path for tree.save()."""
    return str(tmp_path / "tree.bin")
