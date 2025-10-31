import dask.bag as db
import dask

# Use single-threaded scheduler to avoid multiprocessing issues
dask.config.set(scheduler='synchronous')

data = [5.0]
b = db.from_sequence(data, npartitions=1)
result = b.var(ddof=1).compute()
print(f"Result: {result}")