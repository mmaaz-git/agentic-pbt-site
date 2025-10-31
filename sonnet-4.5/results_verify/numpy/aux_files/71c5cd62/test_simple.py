import numpy as np

print("Testing np.rec.fromrecords with empty list:")
try:
    rec = np.rec.fromrecords([], names='x,y')
    print(f"Success! Created record array: {rec}")
except IndexError as e:
    print(f"IndexError: {e}")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")

print("\nTesting np.rec.fromrecords with non-empty list:")
try:
    rec = np.rec.fromrecords([(1, 2.0), (3, 4.0)], names='x,y')
    print(f"Success! Created record array with {len(rec)} records")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")