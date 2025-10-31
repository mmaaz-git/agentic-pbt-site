import numpy.char as char
from hypothesis import given, strategies as st, settings, assume

@given(st.text(min_size=0, max_size=30))
@settings(max_examples=100)
def test_find_matches_python_for_null_bytes(s):
    assume('\x00' not in s)

    py_result = s.find('\x00')
    np_result = int(char.find(s, '\x00'))

    assert py_result == np_result, f"find({repr(s)}, '\\x00'): Python={py_result}, NumPy={np_result}"

if __name__ == "__main__":
    print("Running hypothesis test...")
    try:
        test_find_matches_python_for_null_bytes()
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")

    print("\nTesting with specific examples:")
    for s in ['', 'test', 'a', '123', 'hello world']:
        if '\x00' not in s:
            py_result = s.find('\x00')
            np_result = int(char.find(s, '\x00'))
            print(f"String {repr(s)}: Python={py_result}, NumPy={np_result}, Match={py_result == np_result}")