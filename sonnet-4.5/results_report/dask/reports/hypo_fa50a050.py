from hypothesis import given, strategies as st
import dask.bag as db
import dask

# Use single-threaded scheduler to avoid multiprocessing issues
dask.config.set(scheduler='synchronous')

@given(st.integers(min_value=1, max_value=10))
def test_variance_division_by_zero(n):
    data = [float(i) for i in range(n)]
    b = db.from_sequence(data, npartitions=1)

    b.var(ddof=n).compute()

if __name__ == "__main__":
    test_variance_division_by_zero()