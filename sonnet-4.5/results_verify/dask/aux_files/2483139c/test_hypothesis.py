import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

import numpy as np
from hypothesis import assume, given, settings, strategies as st
from dask.array.slicing import normalize_slice


@given(
    st.integers(min_value=-20, max_value=20),
    st.integers(min_value=-20, max_value=20),
    st.integers(min_value=-10, max_value=10).filter(lambda x: x != 0),
    st.integers(min_value=1, max_value=100)
)
@settings(max_examples=500)
def test_normalize_slice_equivalence(start, stop, step, dim):
    assume(dim > 0)
    arr = np.arange(dim)
    idx = slice(start, stop, step)

    normalized = normalize_slice(idx, dim)

    original_result = arr[idx]
    normalized_result = arr[normalized]

    assert np.array_equal(original_result, normalized_result), \
        f"Normalized slice should produce same elements: {idx} vs {normalized} on array of length {dim}"

# Test the specific failing case
def test_specific_failure():
    start, stop, step, dim = -2, -2, -1, 1
    arr = np.arange(dim)
    idx = slice(start, stop, step)

    normalized = normalize_slice(idx, dim)

    original_result = arr[idx]
    normalized_result = arr[normalized]

    print(f"Testing: start={start}, stop={stop}, step={step}, dim={dim}")
    print(f"Array: {arr}")
    print(f"Original slice: {idx}")
    print(f"Normalized slice: {normalized}")
    print(f"Original result: {original_result}")
    print(f"Normalized result: {normalized_result}")
    print(f"Equal: {np.array_equal(original_result, normalized_result)}")

    assert np.array_equal(original_result, normalized_result), \
        f"Normalized slice should produce same elements: {idx} vs {normalized} on array of length {dim}"

if __name__ == "__main__":
    # First test the specific failing case
    print("Testing specific failure case...")
    try:
        test_specific_failure()
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")

    print("\n" + "="*50 + "\n")

    # Run the property-based test
    print("Running property-based test...")
    try:
        test_normalize_slice_equivalence()
        print("All property tests passed!")
    except Exception as e:
        print(f"Property test failed: {e}")