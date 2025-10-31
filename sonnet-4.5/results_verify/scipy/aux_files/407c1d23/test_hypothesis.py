import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env')

import numpy as np
import scipy.integrate as integrate
from hypothesis import given, strategies as st, settings, example

@given(
    st.lists(st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False),
             min_size=3, max_size=100)
)
@settings(max_examples=100)  # Reduced for testing
@example([0.0, 0.0, 1.0])  # The specific failing case
def test_cumulative_simpson_monotonic_for_positive(y_list):
    y = np.array(y_list)
    cumulative = integrate.cumulative_simpson(y, initial=0)

    for i in range(len(cumulative) - 1):
        if cumulative[i] > cumulative[i+1]:
            print(f"Monotonicity violated!")
            print(f"Input: {y}")
            print(f"Cumulative: {cumulative}")
            print(f"At index {i}: {cumulative[i]} > {cumulative[i+1]}")
            assert False, f"Monotonicity violated at index {i}: {cumulative[i]} > {cumulative[i+1]}"

# Run the test
print("Running property-based test...")
try:
    test_cumulative_simpson_monotonic_for_positive()
    print("Test passed for all examples")
except AssertionError as e:
    print(f"Test failed: {e}")
except Exception as e:
    print(f"Test error: {e}")