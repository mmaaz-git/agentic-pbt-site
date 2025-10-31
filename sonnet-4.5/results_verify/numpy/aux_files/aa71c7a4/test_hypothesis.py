import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st


@given(st.lists(st.text(), min_size=1, max_size=10))
def test_operations_on_arrays(strings):
    arr = np.array(strings)
    result = nps.strip(arr)
    assert len(result) == len(arr)
    for i, (original, stripped) in enumerate(zip(strings, result)):
        assert stripped == original.strip()

# Run the test with the specific failing input manually
def test_manual():
    strings = ['\x00']
    arr = np.array(strings)
    result = nps.strip(arr)
    assert len(result) == len(arr)
    for i, (original, stripped) in enumerate(zip(strings, result)):
        assert stripped == original.strip()

try:
    test_manual()
    print("Test passed with ['\x00']")
except AssertionError as e:
    print(f"Test failed with ['\x00']: {e}")

# Also run general hypothesis testing
if __name__ == "__main__":
    print("\nRunning Hypothesis tests...")
    try:
        test_operations_on_arrays()
        print("All Hypothesis tests passed")
    except Exception as e:
        print(f"Hypothesis test failed: {e}")