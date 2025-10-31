import warnings
import numpy as np
import traceback

print("Test 1: Basic reproduction test")
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PendingDeprecationWarning)

        A = np.matrix([[1, 2]])
        result = np.bmat('A', gdict={'A': A})
        print(f"Success! Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n\nTest 2: Property-based test case")
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PendingDeprecationWarning)

        rows, cols = 1, 1
        globals_dict = {
            'A': np.matrix(np.ones((rows, cols))),
            'B': np.matrix(np.zeros((rows, cols)))
        }

        result = np.bmat('A, B', gdict=globals_dict)
        print(f"Success! Result shape: {result.shape}")
        print(f"Result:\n{result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n\nTest 3: With both ldict and gdict")
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PendingDeprecationWarning)

        A = np.matrix([[1, 2]])
        B = np.matrix([[3, 4]])
        result = np.bmat('A, B', ldict={'A': A}, gdict={'B': B})
        print(f"Success! Result:\n{result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n\nTest 4: With only ldict")
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PendingDeprecationWarning)

        A = np.matrix([[1, 2]])
        result = np.bmat('A', ldict={'A': A}, gdict={})
        print(f"Success! Result:\n{result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n\nTest 5: Check what happens in _from_string with ldict=None")
print("Simulating _from_string behavior:")
ldict = None
col = 'A'
try:
    thismat = ldict[col]
    print(f"Success accessing ldict[{col}]")
except TypeError as e:
    print(f"TypeError when ldict is None: {e}")
except KeyError as e:
    print(f"KeyError when ldict is None: {e}")