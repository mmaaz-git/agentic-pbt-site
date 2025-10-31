from hypothesis import given, strategies as st, settings
import numpy.ctypeslib
import numpy as np


@given(st.tuples(st.integers(min_value=-10, max_value=-1), st.integers(min_value=1, max_value=10)))
@settings(max_examples=200)
def test_ndpointer_negative_shape(shape):
    try:
        ptr = numpy.ctypeslib.ndpointer(shape=shape)
        print(f"Created ndpointer with negative shape: {shape}")
        print(f"  Pointer: {ptr}")
        print(f"  Shape attribute: {ptr._shape_}")
        assert False, f"Should reject shape with negative dimensions {shape}"
    except (TypeError, ValueError):
        pass

# Run the test
if __name__ == "__main__":
    # Test with specific failing inputs mentioned in the report
    test_cases = [(-1, 3), (0, -1), (-5, 10)]

    for shape in test_cases:
        print(f"\nTesting shape: {shape}")
        try:
            ptr = numpy.ctypeslib.ndpointer(shape=shape)
            print(f"  Created ndpointer successfully!")
            print(f"  Pointer: {ptr}")
            print(f"  Shape attribute: {ptr._shape_}")
            print(f"  ERROR: Should have rejected shape with negative dimensions {shape}")
        except (TypeError, ValueError) as e:
            print(f"  Correctly rejected: {e}")