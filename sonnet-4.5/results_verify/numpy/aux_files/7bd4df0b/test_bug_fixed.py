import numpy as np
import traceback

print("Testing bug reproduction...")
print("=" * 50)

# Test 1: Basic reproduction as described in bug report
print("\nTest 1: Basic reproduction with gdict only")
try:
    A = np.matrix([[1, 2], [3, 4]])
    B = np.matrix([[5, 6], [7, 8]])
    result = np.bmat('A,B', gdict={'A': A, 'B': B})
    print(f"Result: {result}")
except TypeError as e:
    print(f"✓ Confirmed bug - Error type: {type(e).__name__}")
    print(f"  Error message: {e}")
    if "'NoneType' object is not subscriptable" in str(e):
        print("  This is exactly the error described in the bug report")

print("\n" + "=" * 50)

# Test 2: Verify it works when both ldict and gdict are provided
print("\nTest 2: With both ldict and gdict")
try:
    A = np.matrix([[1, 2], [3, 4]])
    B = np.matrix([[5, 6], [7, 8]])
    result = np.bmat('A,B', ldict={'A': A, 'B': B}, gdict={})
    print(f"✓ Works correctly - Result:\n{result}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 50)

# Test 3: With ldict as empty dict and gdict with variables
print("\nTest 3: With empty ldict and gdict with variables")
try:
    A = np.matrix([[1, 2], [3, 4]])
    B = np.matrix([[5, 6], [7, 8]])
    result = np.bmat('A,B', ldict={}, gdict={'A': A, 'B': B})
    print(f"✓ Works correctly - Result:\n{result}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 50)

# Test 4: Normal case - should fallback to gdict when ldict doesn't have variable
print("\nTest 4: Variables in gdict should be found when ldict is empty")
try:
    X = np.matrix([[1, 2], [3, 4]])
    Y = np.matrix([[5, 6], [7, 8]])
    # X is in ldict, Y is in gdict - should work
    result = np.bmat('X,Y', ldict={'X': X}, gdict={'Y': Y})
    print(f"✓ Works correctly - Result:\n{result}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 50)

# Test 5: Check the proposed fix
print("\nTest 5: Testing the proposed fix")
print("The bug occurs because when gdict is provided without ldict,")
print("the code sets loc_dict = ldict (which is None)")
print("Then _from_string tries to do ldict[col] which fails with TypeError")
print("")
print("The proposed fix would set loc_dict = {} when ldict is None")
print("This would allow the KeyError to be caught and fallback to gdict")