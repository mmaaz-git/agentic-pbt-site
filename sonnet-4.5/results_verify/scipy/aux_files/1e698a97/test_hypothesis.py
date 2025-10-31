import math
import numpy as np
from hypothesis import given, settings, strategies as st
from scipy.spatial import distance

@settings(max_examples=1000)
@given(n=st.integers(1, 20))
def test_dice_all_zeros(n):
    x = np.zeros(n, dtype=bool)
    y = np.zeros(n, dtype=bool)

    dist = distance.dice(x, y)

    assert math.isclose(dist, 0.0, abs_tol=1e-9)

if __name__ == "__main__":
    # Run the property-based test
    test_dice_all_zeros()
    print("All tests passed!")