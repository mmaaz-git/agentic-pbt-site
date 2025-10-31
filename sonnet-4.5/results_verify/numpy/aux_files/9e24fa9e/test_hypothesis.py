from hypothesis import given, strategies as st, example
import numpy as np
import numpy.strings as nps

@given(st.lists(st.text(min_size=1), min_size=1, max_size=10), st.text(min_size=1, max_size=5))
@example(strings=['abc'], new='X')  # The failing example from the bug report
def test_replace_matches_python(strings, new):
    old = '\x00'  # Focus specifically on null character
    for s in strings:
        arr = np.array([s])
        np_result = nps.replace(arr, old, new)[0]
        py_result = s.replace(old, new)
        if np_result != py_result:
            print(f"FAILURE: String={repr(s)}, old={repr(old)}, new={repr(new)}")
            print(f"  Python: {repr(py_result)}")
            print(f"  NumPy:  {repr(np_result)}")
            assert False, f"Mismatch for {repr(s)}"

# Run the test
try:
    test_replace_matches_python()
    print("Hypothesis test passed (which means the bug is NOT present)")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")
    print("This confirms the bug exists")