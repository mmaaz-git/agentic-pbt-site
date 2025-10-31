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
    # Test with the specific failing input
    u_list = [0.0, 0.0, 0.0]
    u = np.array(u_list)
    d = braycurtis(u, u)
    print(f"Testing with u_list={u_list}")
    print(f"braycurtis(u, u) = {d}")
    print(f"Is it NaN? {np.isnan(d)}")
    print(f"Is it close to 0? {np.isclose(d, 0.0) if not np.isnan(d) else False}")

    # Run the hypothesis test
    print("\nRunning Hypothesis test...")
    test_braycurtis_identity()