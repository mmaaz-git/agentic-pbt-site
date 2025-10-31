from hypothesis import given, strategies as st
from pandas.core.util.hashing import hash_tuples
import numpy as np
import pytest


def test_hash_tuples_empty():
    """Test that hash_tuples handles empty list input gracefully."""
    hashed = hash_tuples([])
    assert len(hashed) == 0
    assert hashed.dtype == np.uint64


if __name__ == "__main__":
    test_hash_tuples_empty()