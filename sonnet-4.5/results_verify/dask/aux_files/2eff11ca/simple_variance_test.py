import dask
import dask.bag as db

dask.config.set(scheduler='synchronous')

print("Testing variance with different ddof values:")
print("=" * 50)

b = db.from_sequence([1.0, 2.0], npartitions=1)

# Test different ddof values
for ddof in range(0, 5):
    try:
        result = b.var(ddof=ddof).compute()
        print(f"ddof={ddof}: variance = {result}")
        if result < 0:
            print(f"  WARNING: Negative variance detected!")
    except Exception as e:
        print(f"ddof={ddof}: ERROR - {e}")

print("\nTesting with larger dataset:")
b2 = db.from_sequence([1.0, 2.0, 3.0, 4.0, 5.0], npartitions=1)
for ddof in range(0, 7):
    try:
        result = b2.var(ddof=ddof).compute()
        print(f"ddof={ddof}: variance = {result}")
        if result < 0:
            print(f"  WARNING: Negative variance detected!")
    except Exception as e:
        print(f"ddof={ddof}: ERROR - {e}")