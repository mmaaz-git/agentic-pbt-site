import dask
import dask.bag as db

dask.config.set(scheduler='synchronous')

# Test case 1: ddof equals n (both are 2)
print("Test case 1: [1.0, 2.0] with ddof=2")
b = db.from_sequence([1.0, 2.0], npartitions=1)
try:
    result = b.var(ddof=2).compute()
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\n" + "="*50 + "\n")

# Test case 2: Empty sequence
print("Test case 2: Empty sequence [] with ddof=0")
b = db.from_sequence([], npartitions=1)
try:
    result = b.var(ddof=0).compute()
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\n" + "="*50 + "\n")

# Test case 3: ddof > n
print("Test case 3: [1.0, 2.0, 3.0] with ddof=4")
b = db.from_sequence([1.0, 2.0, 3.0], npartitions=1)
try:
    result = b.var(ddof=4).compute()
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")