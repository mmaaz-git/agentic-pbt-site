from hypothesis import given, strategies as st, example
from pandas.core.indexers import length_of_indexer

@given(
    start=st.integers(min_value=0, max_value=100),
    stop=st.integers(min_value=0, max_value=100),
    step=st.integers(min_value=1, max_value=10)
)
@example(start=1, stop=0, step=1)  # The specific failing case mentioned
def test_length_of_indexer_range_consistency(start, stop, step):
    rng = range(start, stop, step)
    expected_length = len(rng)
    predicted_length = length_of_indexer(rng)

    print(f"Testing range({start}, {stop}, {step}): expected={expected_length}, predicted={predicted_length}")
    assert expected_length == predicted_length, f"Expected {expected_length}, got {predicted_length} for range({start}, {stop}, {step})"

# Run the test
if __name__ == "__main__":
    test_length_of_indexer_range_consistency()