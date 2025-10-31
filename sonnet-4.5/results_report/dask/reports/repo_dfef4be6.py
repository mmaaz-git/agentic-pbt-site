import dask
import dask.bag as db

dask.config.set(scheduler='synchronous')

b = db.from_sequence([1.0, 2.0], npartitions=1)
result = b.var(ddof=3).compute()
print(f"Variance: {result}")
print(f"Expected: Variance should be non-negative, but got {result}")