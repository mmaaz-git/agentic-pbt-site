#!/usr/bin/env python3
"""Test the reported bug in CyLocals.invoke"""

from hypothesis import given, strategies as st

sortkey = lambda item: item[0].lower()

def cy_locals_invoke_logic(local_cython_vars):
    max_name_length = len(max(local_cython_vars, key=len))
    for name, cyvar in sorted(local_cython_vars.items(), key=sortkey):
        pass
    return max_name_length

# Test 1: Property-based test
@given(st.dictionaries(st.text(min_size=1), st.integers()))
def test_cy_locals_with_various_dicts(local_vars):
    if len(local_vars) == 0:
        try:
            cy_locals_invoke_logic(local_vars)
            assert False, "Should have raised ValueError for empty dict"
        except ValueError:
            pass
    else:
        result = cy_locals_invoke_logic(local_vars)
        assert result >= 0

# Test 2: Direct reproduction
def test_direct_reproduction():
    print("Testing direct reproduction of the bug...")
    local_cython_vars = {}

    try:
        max_name_length = len(max(local_cython_vars, key=len))
        print(f"ERROR: Should have raised ValueError, got {max_name_length}")
    except ValueError as e:
        print(f"✓ Got expected ValueError: {e}")

    # Test with non-empty dict
    local_cython_vars = {"test": 1, "another": 2}
    try:
        max_name_length = len(max(local_cython_vars, key=len))
        print(f"✓ With non-empty dict, max_name_length = {max_name_length}")
    except Exception as e:
        print(f"ERROR: Unexpected exception with non-empty dict: {e}")

if __name__ == "__main__":
    # Run direct test
    test_direct_reproduction()

    # Run hypothesis test
    print("\nRunning property-based test...")
    test_cy_locals_with_various_dicts()
    print("✓ Property-based test passed")