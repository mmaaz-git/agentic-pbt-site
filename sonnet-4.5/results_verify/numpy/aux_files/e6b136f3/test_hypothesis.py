import numpy.char as char
import numpy as np
from hypothesis import given, strategies as st, settings


st_text = st.text(
    alphabet=st.characters(blacklist_categories=('Cs',), blacklist_characters='\x00'),
    min_size=0,
    max_size=20
)

def check_upper(arr):
    result = char.upper(arr)
    for i in range(len(arr)):
        assert result[i] == arr[i].upper(), f"Failed on {arr[i]!r}: numpy={result[i]!r}, python={arr[i].upper()!r}"

@given(st.lists(st_text, min_size=1, max_size=10).map(lambda lst: np.array(lst, dtype='U')))
@settings(max_examples=100)
def test_upper_matches_python(arr):
    check_upper(arr)

# Test with the specific failing input
if __name__ == "__main__":
    # Test with the specific failing case
    test_arr = np.array(['ß'])
    print(f"Testing with {test_arr}")
    try:
        check_upper(test_arr)
        print("Test passed")
    except AssertionError as e:
        print(f"Test failed: {e}")

    # Also run the hypothesis test
    print("\nRunning hypothesis test...")
    test_upper_matches_python()