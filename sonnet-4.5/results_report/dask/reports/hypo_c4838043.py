from hypothesis import given, strategies as st
import dask.bag as db
import dask

dask.config.set(scheduler='synchronous')

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100), min_size=0, max_size=10))
def test_variance_no_crash(data):
    b = db.from_sequence(data if data else [0], npartitions=1)
    for ddof in range(0, len(data) + 2):
        try:
            var = b.var(ddof=ddof).compute()
        except ZeroDivisionError:
            assert False, f"var() should not crash with ZeroDivisionError for ddof={ddof}, n={len(data)}"

if __name__ == "__main__":
    test_variance_no_crash()