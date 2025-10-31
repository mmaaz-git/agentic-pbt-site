import numpy as np
from xarray.testing import assert_duckarray_equal

# Test case: 0-dimensional arrays with different values
x = np.array(5.0)
y = np.array(3.0)

print(f"x shape: {x.shape}, x value: {x}")
print(f"y shape: {y.shape}, y value: {y}")
print("Attempting to compare 0-dimensional arrays with different values...")

try:
    assert_duckarray_equal(x, y)
except TypeError as e:
    print(f"\nTypeError occurred: {e}")
    print(f"Error type: {type(e).__name__}")
except AssertionError as e:
    print(f"\nAssertionError (expected): {e}")