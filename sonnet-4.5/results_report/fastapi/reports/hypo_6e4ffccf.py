from hypothesis import given, strategies as st
import dask.bag as db
import dask

dask.config.set(scheduler='synchronous')

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100), min_size=0, max_size=100))
def test_mean_no_crash(data):
    b = db.from_sequence(data, npartitions=1)
    try:
        mean = b.mean().compute()
        # If we get here, the mean was computed successfully
        if len(data) > 0:
            # Basic sanity check that mean is within range of data
            assert min(data) <= mean <= max(data)
    except ZeroDivisionError:
        # This should only happen if the bag was empty
        assert len(data) == 0, f"mean() raised ZeroDivisionError on non-empty sequence with {len(data)} elements"
        # But actually, this is the bug - mean() should handle empty sequences gracefully
        raise AssertionError("mean() should not crash with ZeroDivisionError on empty sequence")

# Run the test
test_mean_no_crash()