import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st, settings, assume

@given(st.lists(st.text(), min_size=1, max_size=5))
@settings(max_examples=500)
def test_add_broadcast(strings):
    arr = np.array(strings, dtype=str)
    scalar = 'test'
    result = nps.add(arr, scalar)
    for i in range(len(arr)):
        expected = strings[i] + scalar
        if result[i] != expected:
            print(f"Failed at index {i}:")
            print(f"  Input: {repr(strings[i])}")
            print(f"  Expected: {repr(expected)}")
            print(f"  Got: {repr(result[i])}")
            raise AssertionError(f"Failed: {repr(result[i])} != {repr(expected)}")

# Test specific failing case
print("Testing specific failing case: ['\\x00']")
try:
    test_add_broadcast(['\x00'])
    print("Test passed (unexpected)")
except AssertionError as e:
    print(f"Test failed as expected: {e}")

# Run full hypothesis test
print("\nRunning full hypothesis test...")
try:
    test_add_broadcast()
    print("All tests passed")
except Exception as e:
    print(f"Test failed: {e}")