from hypothesis import given, strategies as st, settings
import numpy as np
import numpy.strings as nps

@given(st.lists(st.text(min_size=1), min_size=1, max_size=10), st.text(min_size=1, max_size=5), st.text(max_size=5))
@settings(max_examples=100)
def test_replace_matches_python(strings, old, new):
    for s in strings:
        if old in s:
            arr = np.array([s])
            np_result = nps.replace(arr, old, new)[0]
            py_result = s.replace(old, new)
            assert np_result == py_result, f"Failed: s='{s}', old='{old}', new='{new}'. Expected '{py_result}', got '{np_result}'"

# Test with the specific failing input
print("Testing with specific failing input: strings=['0'], old='0', new='00'")
try:
    strings = ['0']
    old = '0'
    new = '00'
    for s in strings:
        if old in s:
            arr = np.array([s])
            np_result = nps.replace(arr, old, new)[0]
            py_result = s.replace(old, new)
            assert np_result == py_result, f"Failed: s='{s}', old='{old}', new='{new}'. Expected '{py_result}', got '{np_result}'"
    print("Test passed")
except AssertionError as e:
    print(f"Test failed: {e}")

# Run the general property test
print("\nRunning property-based test with random inputs...")
try:
    test_replace_matches_python()
    print("All tests passed")
except Exception as e:
    print(f"Test failed: {e}")