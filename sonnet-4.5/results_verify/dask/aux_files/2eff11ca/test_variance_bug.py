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

# Test with the specific failing input
if __name__ == "__main__":
    # First test the specific failing case
    print("Testing specific failing case: [1.0, 2.0] with ddof=3")
    b = db.from_sequence([1.0, 2.0], npartitions=1)
    result = b.var(ddof=3).compute()
    print(f"Variance with ddof=3: {result}")
    print(f"Is variance negative? {result < 0}")

    # Run the property-based test
    print("\nRunning property-based test...")
    try:
        test_variance_non_negative()
    except AssertionError as e:
        print(f"Property test failed: {e}")