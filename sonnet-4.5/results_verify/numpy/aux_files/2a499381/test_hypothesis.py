import numpy as np
import numpy.ma as ma
from hypothesis import given, settings, seed, example
import hypothesis.extra.numpy as npst
import hypothesis.strategies as st

@given(npst.arrays(dtype=np.float64, shape=npst.array_shapes(max_dims=2, max_side=20)),
       npst.arrays(dtype=np.float64, shape=npst.array_shapes(max_dims=2, max_side=20)))
@settings(max_examples=500)
@seed(1234)  # For reproducibility
def test_allclose_symmetry(data1, data2):
    if data1.shape != data2.shape:
        return

    mask1 = np.random.rand(*data1.shape) < 0.3 if data1.size > 0 else np.array([])
    mask2 = np.random.rand(*data2.shape) < 0.3 if data2.size > 0 else np.array([])

    x = ma.array(data1, mask=mask1)
    y = ma.array(data2, mask=mask2)

    result_xy = ma.allclose(x, y)
    result_yx = ma.allclose(y, x)

    if result_xy != result_yx:
        print(f"\nFound asymmetric case:")
        print(f"  x data: {data1}")
        print(f"  x mask: {mask1}")
        print(f"  y data: {data2}")
        print(f"  y mask: {mask2}")
        print(f"  allclose(x, y) = {result_xy}")
        print(f"  allclose(y, x) = {result_yx}")

    assert result_xy == result_yx, f"allclose should be symmetric. Got allclose(x,y)={result_xy}, allclose(y,x)={result_yx}"

# Run the test
try:
    test_allclose_symmetry()
    print("Property-based test completed - no failures found in random examples")
except AssertionError as e:
    print(f"Property-based test FAILED: {e}")

# Also test the specific case mentioned in the bug report
print("\nTesting specific infinity case:")
x = ma.array([np.inf], mask=[False])
y = ma.array([0.], mask=[True])
try:
    assert ma.allclose(x, y) == ma.allclose(y, x), "Specific infinity case failed"
    print("Specific test passed")
except AssertionError:
    print(f"Specific test FAILED: allclose(x,y)={ma.allclose(x, y)}, allclose(y,x)={ma.allclose(y, x)}")