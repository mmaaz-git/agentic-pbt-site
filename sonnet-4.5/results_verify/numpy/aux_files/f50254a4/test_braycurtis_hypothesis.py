import numpy as np
from hypothesis import given, strategies as st, settings
from scipy.spatial.distance import braycurtis


@given(
    st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=1, max_size=100)
)
@settings(max_examples=500)
def test_braycurtis_identity(u_list):
    u = np.array(u_list)
    d = braycurtis(u, u)
    assert np.isclose(d, 0.0), f"braycurtis(u, u) should be 0, got {d}"

if __name__ == "__main__":
    # Test with the failing input directly
    u_list = [0.0, 0.0, 0.0]
    print(f"Testing with u_list={u_list}")
    u = np.array(u_list)
    d = braycurtis(u, u)
    print(f"braycurtis(u, u) = {d}")
    if np.isnan(d):
        print(f"Test FAILED: Expected 0.0, got nan")
    elif not np.isclose(d, 0.0):
        print(f"Test FAILED: braycurtis(u, u) should be 0, got {d}")
    else:
        print("Test passed")

    # Run the full Hypothesis test
    print("\nRunning full Hypothesis test...")
    test_braycurtis_identity()