import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st, assume


def string_arrays():
    return st.lists(
        st.text(
            alphabet=st.characters(min_codepoint=32, max_codepoint=126),
            min_size=0, max_size=30
        ),
        min_size=1, max_size=20
    ).map(lambda lst: np.array(lst, dtype='U'))


def simple_strings():
    return st.text(
        alphabet=st.characters(min_codepoint=32, max_codepoint=126),
        min_size=0, max_size=20
    )


@given(string_arrays(), simple_strings(), simple_strings(), st.integers(min_value=0, max_value=10))
def test_replace_matches_python_semantics(arr, old_str, new_str, count):
    assume(old_str != '')

    result = nps.replace(arr, old_str, new_str, count=count)

    for i in range(len(arr)):
        expected = str(arr[i]).replace(old_str, new_str, count)
        actual = str(result[i])
        assert actual == expected, f"Failed for arr[{i}]='{arr[i]}', old='{old_str}', new='{new_str}', count={count}. Expected '{expected}', got '{actual}'"


if __name__ == "__main__":
    # Test with the specific failing example
    arr = np.array(['0'])
    old_str = '0'
    new_str = '00'
    count = 1

    print(f"Testing with arr={arr}, old_str='{old_str}', new_str='{new_str}', count={count}")

    result = nps.replace(arr, old_str, new_str, count=count)
    expected = str(arr[0]).replace(old_str, new_str, count)
    actual = str(result[0])

    print(f"Expected: '{expected}'")
    print(f"Actual: '{actual}'")
    print(f"Test {'PASSED' if actual == expected else 'FAILED'}")

    # Run hypothesis test
    print("\nRunning hypothesis tests...")
    try:
        test_replace_matches_python_semantics()
    except AssertionError as e:
        print(f"Test failed: {e}")