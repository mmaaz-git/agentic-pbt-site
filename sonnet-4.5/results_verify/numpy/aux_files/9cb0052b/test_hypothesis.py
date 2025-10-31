import numpy.rec
import math
from hypothesis import given, strategies as st


@given(st.lists(st.floats(allow_nan=True, allow_infinity=False), min_size=0, max_size=20))
def test_find_duplicate_nan_behavior(lst):
    result = numpy.rec.find_duplicate(lst)
    nan_count = sum(1 for x in lst if isinstance(x, float) and math.isnan(x))

    if nan_count > 1:
        nan_in_result = any(isinstance(x, float) and math.isnan(x) for x in result)
        assert nan_in_result, f"Multiple NaN values in input but none in result. Input: {lst}, Result: {result}"

if __name__ == "__main__":
    # Test with the specific failing case manually
    lst = [float('nan'), float('nan'), 1.0, 1.0]
    result = numpy.rec.find_duplicate(lst)
    nan_count = sum(1 for x in lst if isinstance(x, float) and math.isnan(x))

    print(f"Testing with input: {lst}")
    print(f"Result: {result}")
    print(f"NaN count in input: {nan_count}")

    if nan_count > 1:
        nan_in_result = any(isinstance(x, float) and math.isnan(x) for x in result)
        print(f"NaN in result: {nan_in_result}")
        assert nan_in_result, f"Multiple NaN values in input but none in result. Input: {lst}, Result: {result}"

    print("\nTest passed with specific case")

    # Run the property-based tests
    from hypothesis import example
    test = test_find_duplicate_nan_behavior
    test = example([float('nan'), float('nan'), 1.0, 1.0])(test)
    test()