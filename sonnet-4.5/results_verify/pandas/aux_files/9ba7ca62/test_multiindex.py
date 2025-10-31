from pandas import MultiIndex
import traceback

print("Testing MultiIndex.from_tuples([]):")
print("=" * 40)

try:
    result = MultiIndex.from_tuples([])
    print(f"Success: {result}")
    print(f"nlevels: {result.nlevels}")
    print(f"Length: {len(result)}")
except TypeError as e:
    print(f"Failed with TypeError: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
except Exception as e:
    print(f"Failed with unexpected error: {type(e).__name__}: {e}")
    print("\nFull traceback:")
    traceback.print_exc()