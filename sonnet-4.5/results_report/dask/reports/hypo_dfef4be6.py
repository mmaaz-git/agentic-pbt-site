from hypothesis import given, strategies as st
import dask.bag as db
import dask

dask.config.set(scheduler='synchronous')

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100), min_size=1, max_size=10))
def test_variance_non_negative(data):
    b = db.from_sequence(data, npartitions=1)
    for ddof in range(0, len(data) + 2):
        var = b.var(ddof=ddof).compute()
        assert var >= 0, f"Variance must be non-negative, got {var} with ddof={ddof}, n={len(data)}"

# Run the test
if __name__ == "__main__":
    test_variance_non_negative()