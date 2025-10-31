import numpy as np
import numpy.strings
from hypothesis import given, strategies as st, settings


@given(st.lists(st.text(), min_size=1), st.text(min_size=1), st.text(min_size=1))
@settings(max_examples=1000)
def test_add_associativity(strings, s1, s2):
    arr = np.array(strings)
    left = numpy.strings.add(numpy.strings.add(arr, s1), s2)
    right = numpy.strings.add(arr, s1 + s2)
    assert np.array_equal(left, right), f"Failed for strings={strings}, s1={repr(s1)}, s2={repr(s2)}"

# Test with the specific failing input mentioned
def test_specific_case():
    strings = ['']
    s1 = '\x00'
    s2 = '0'

    arr = np.array(strings)
    left = numpy.strings.add(numpy.strings.add(arr, s1), s2)
    right = numpy.strings.add(arr, s1 + s2)

    print(f"Testing with: strings={strings}, s1={repr(s1)}, s2={repr(s2)}")
    print(f"Left result (add(add(arr, s1), s2)): {repr(left[0])}")
    print(f"Right result (add(arr, s1+s2)): {repr(right[0])}")
    print(f"Are they equal? {np.array_equal(left, right)}")

    assert np.array_equal(left, right), f"Associativity violation found!"

if __name__ == "__main__":
    # First test the specific case
    print("Testing the specific case mentioned in the bug report:")
    print("-" * 50)
    try:
        test_specific_case()
        print("Specific case passed!")
    except AssertionError as e:
        print(f"Specific case failed: {e}")

    print("\n" + "=" * 50)
    print("Running Hypothesis test to find more failures:")
    print("-" * 50)

    # Run the hypothesis test
    test_add_associativity()