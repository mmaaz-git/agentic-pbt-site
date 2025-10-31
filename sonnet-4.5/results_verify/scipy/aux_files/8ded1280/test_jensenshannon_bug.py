import numpy as np
from hypothesis import given, strategies as st, settings
from scipy.spatial.distance import jensenshannon

# Property-based test from the bug report
@given(
    st.lists(st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=1, max_size=100)
)
@settings(max_examples=500)
def test_jensenshannon_identity(p_list):
    p = np.array(p_list)
    d = jensenshannon(p, p)
    assert np.isclose(d, 0.0), f"jensenshannon(p, p) should be 0, got {d}"

# Run the test
print("Running property-based test...")
try:
    test_jensenshannon_identity()
    print("Property test passed")
except AssertionError as e:
    print(f"Property test failed: {e}")
except Exception as e:
    print(f"Property test error: {e}")

# Specific reproduction case
print("\nTesting specific failing input [0.0, 0.0, 0.0]...")
p = np.array([0.0, 0.0, 0.0])
result = jensenshannon(p, p)
print(f"jensenshannon([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]) = {result}")
print(f"Is result nan? {np.isnan(result)}")

# Also test with q different from p (both all zeros)
q = np.array([0.0, 0.0, 0.0])
result2 = jensenshannon(p, q)
print(f"\njensenshannon(p=[0.0, 0.0, 0.0], q=[0.0, 0.0, 0.0]) = {result2}")
print(f"Is result nan? {np.isnan(result2)}")

# Test with valid probability distributions for comparison
print("\nFor comparison, testing with valid distributions:")
p_valid = np.array([0.3, 0.3, 0.4])
result3 = jensenshannon(p_valid, p_valid)
print(f"jensenshannon([0.3, 0.3, 0.4], [0.3, 0.3, 0.4]) = {result3}")
print(f"Is close to 0? {np.isclose(result3, 0.0)}")