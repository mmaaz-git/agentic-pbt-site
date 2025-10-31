#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from hypothesis import given, strategies as st, settings
from pandas.core.computation.common import ensure_decoded

@given(st.binary(min_size=0, max_size=100))
@settings(max_examples=100, deadline=None)
def test_ensure_decoded_bytes_to_str(b):
    try:
        result = ensure_decoded(b)
        assert isinstance(result, str)
        print(f"✓ Passed for bytes: {b!r}")
    except Exception as e:
        print(f"✗ Failed for bytes: {b!r}")
        print(f"  Exception: {type(e).__name__}: {e}")
        raise

if __name__ == "__main__":
    print("Running Hypothesis test...")
    try:
        test_ensure_decoded_bytes_to_str()
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\nTest failed!")
        print(f"Final exception: {e}")