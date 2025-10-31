import numpy.rec as rec

# Test case 1: Empty list with formats
try:
    print("Test 1: rec.array([], formats=['i4'], names='x')")
    r = rec.array([], formats=['i4'], names='x')
    print(f"Success: Created array with shape {r.shape}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print()

# Test case 2: Empty tuple with formats
try:
    print("Test 2: rec.array((), formats=['i4'], names='x')")
    r = rec.array((), formats=['i4'], names='x')
    print(f"Success: Created array with shape {r.shape}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print()

# Test case 3: Empty list with multiple fields
try:
    print("Test 3: rec.array([], formats=['i4', 'i4'], names='x,y')")
    r = rec.array([], formats=['i4', 'i4'], names='x,y')
    print(f"Success: Created array with shape {r.shape}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")