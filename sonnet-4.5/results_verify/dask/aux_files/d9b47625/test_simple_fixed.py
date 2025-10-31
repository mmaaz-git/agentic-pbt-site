import dask
import dask.bag as db
import traceback

# Use synchronous scheduler to avoid multiprocessing issues
dask.config.set(scheduler='synchronous')

print("Testing simple reproduction case from bug report...")
print("=" * 50)

# Test case from bug report
data = [5.0]
b = db.from_sequence(data, npartitions=1)

print(f"Data: {data}")
print(f"Number of elements: {len(data)}")
print(f"Using ddof=1 (sample variance)")

try:
    result = b.var(ddof=1).compute()
    print(f"Result: {result}")
except ZeroDivisionError as e:
    print(f"\nError occurred as expected!")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
except Exception as e:
    print(f"\nUnexpected error occurred!")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    traceback.print_exc()

print("\n" + "=" * 50)
print("\nTesting with ddof=0 (population variance)...")

try:
    result = b.var(ddof=0).compute()
    print(f"Result with ddof=0: {result}")
except Exception as e:
    print(f"Error with ddof=0: {type(e).__name__}: {e}")

print("\n" + "=" * 50)
print("\nTesting with 2 elements and ddof=2...")

data2 = [1.0, 2.0]
b2 = db.from_sequence(data2, npartitions=1)

try:
    result = b2.var(ddof=2).compute()
    print(f"Result: {result}")
except ZeroDivisionError as e:
    print(f"Error occurred as expected!")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")

print("\n" + "=" * 50)
print("\nLet's check the actual source code location...")
import dask.bag.chunk as chunk
print(f"Location of var_aggregate: {chunk.__file__}")
print(f"Line 36 in var_aggregate: return result * n / (n - ddof)")