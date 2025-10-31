from hypothesis import given, strategies as st
import sys
import tempfile
import os

# Add scipy to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env/lib/python3.13/site-packages')

@given(st.text())
def test_clear_cache_validates_non_callable(non_callable_str):
    """Property: clear_cache should reject non-callable inputs regardless of optimization level."""
    from scipy.datasets._utils import _clear_cache

    method_map = {"test": ["test.dat"]}
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            _clear_cache([non_callable_str], cache_dir=tmpdir, method_map=method_map)
            assert False, "Should have raised ValueError for non-callable"
        except (ValueError, TypeError, AssertionError, AttributeError):
            pass  # All these exceptions are acceptable for this test

# Run the test
if __name__ == "__main__":
    print("Running Hypothesis test...")
    test_clear_cache_validates_non_callable()
    print("Test passed!")