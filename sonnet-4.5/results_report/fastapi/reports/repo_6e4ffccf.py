import dask
import dask.bag as db

dask.config.set(scheduler='synchronous')

# Test case: computing mean of an empty bag
b = db.from_sequence([], npartitions=1)
result = b.mean().compute()
print(f"Result: {result}")