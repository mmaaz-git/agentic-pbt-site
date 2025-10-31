from hypothesis import given, strategies as st, example
import numpy as np
import numpy.strings as nps

@given(st.lists(st.text(), min_size=1, max_size=10))
@example(['\x00'])  # Specifically test the failing case
def test_upper_matches_python(strings):
    for s in strings:
        arr = np.array([s])
        np_result = nps.upper(arr)[0]
        py_result = s.upper()
        try:
            assert np_result == py_result
            print(f"✓ Pass: {repr(s)}")
        except AssertionError:
            print(f"✗ Fail: {repr(s)}")
            print(f"  Python: {repr(py_result)}")
            print(f"  NumPy:  {repr(np_result)}")
            raise

@given(st.lists(st.text(), min_size=1, max_size=10))
@example(['\x00'])  # Specifically test the failing case
def test_lower_matches_python(strings):
    for s in strings:
        arr = np.array([s])
        np_result = nps.lower(arr)[0]
        py_result = s.lower()
        try:
            assert np_result == py_result
            print(f"✓ Pass: {repr(s)}")
        except AssertionError:
            print(f"✗ Fail: {repr(s)}")
            print(f"  Python: {repr(py_result)}")
            print(f"  NumPy:  {repr(np_result)}")
            raise

if __name__ == "__main__":
    print("Testing upper():")
    try:
        test_upper_matches_python()
        print("All upper() tests passed")
    except AssertionError as e:
        print(f"upper() test failed")

    print("\nTesting lower():")
    try:
        test_lower_matches_python()
        print("All lower() tests passed")
    except AssertionError as e:
        print(f"lower() test failed")