import numpy as np
from hypothesis import given, settings, assume
from hypothesis.extra import numpy as npst
from hypothesis import strategies as st
from scipy.cluster.vq import whiten


@given(npst.arrays(dtype=np.float64,
                   shape=npst.array_shapes(min_dims=2, max_dims=2,
                                          min_side=2, max_side=100),
                   elements=st.floats(min_value=-1e6, max_value=1e6,
                                     allow_nan=False, allow_infinity=False)))
@settings(max_examples=500)
def test_whiten_unit_variance(obs):
    std_devs = np.std(obs, axis=0)
    assume(np.all(std_devs > 1e-10))

    result = whiten(obs)

    result_std = np.std(result, axis=0)

    for i, s in enumerate(result_std):
        assert abs(s - 1.0) < 1e-6, f"Column {i} has std {s}, expected 1.0"

# Run the test
print("Running hypothesis test...")
try:
    test_whiten_unit_variance()
    print("Test passed!")
except AssertionError as e:
    print(f"Test failed with error: {e}")
except Exception as e:
    print(f"Test failed with exception: {e}")

# Now test with the specific failing input
print("\nTesting with the specific failing case...")
obs_fail = np.array([[93206.82233024, 93206.82233024]] * 40)
try:
    test_whiten_unit_variance(obs_fail)
    print("Specific test passed (should not happen if bug exists)")
except AssertionError as e:
    print(f"Specific test failed as expected: {e}")
except Exception as e:
    print(f"Specific test failed with exception: {e}")