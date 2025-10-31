import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from hypothesis import given, strategies as st
from pandas.core.util.hashing import hash_tuples
import numpy as np
import pytest

def test_hash_tuples_empty():
    try:
        hashed = hash_tuples([])
        assert len(hashed) == 0
        assert hashed.dtype == np.uint64
        print("Test PASSED: hash_tuples([]) handled empty list correctly")
    except Exception as e:
        print(f"Test FAILED: {type(e).__name__}: {e}")

test_hash_tuples_empty()