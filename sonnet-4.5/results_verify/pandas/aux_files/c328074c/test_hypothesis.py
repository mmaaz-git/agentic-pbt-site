from hypothesis import given, settings, strategies as st, assume
import pandas as pd


@given(
    values=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=2, max_size=50),
    bins=st.integers(min_value=2, max_value=10)
)
@settings(max_examples=200)
def test_cut_assigns_all_values(values, bins):
    assume(len(set(values)) > 1)
    result = pd.cut(values, bins=bins)
    assert len(result) == len(values)


if __name__ == "__main__":
    # Test with the specific failing input
    values = [2.2250738585e-313, -1.0]
    bins = 2
    print(f"Testing with values={values}, bins={bins}")
    try:
        test_cut_assigns_all_values()
    except Exception as e:
        print(f"Hypothesis test failed: {e}")