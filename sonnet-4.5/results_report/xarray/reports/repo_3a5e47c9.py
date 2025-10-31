from xarray.core.utils import OrderedSet

# Create an OrderedSet with some initial values
os = OrderedSet([1, 2, 3])
print(f"Initial OrderedSet: {os}")

# Try to discard an element that doesn't exist
# According to the MutableSet contract, this should NOT raise an error
print("\nAttempting to discard(999) - an element not in the set...")
try:
    os.discard(999)
    print("Success: discard(999) completed without error")
except KeyError as e:
    print(f"ERROR: KeyError raised: {e}")

# For comparison, test with built-in set
print("\n--- Comparison with built-in set ---")
s = {1, 2, 3}
print(f"Initial set: {s}")
print("Attempting to discard(999) - an element not in the set...")
try:
    s.discard(999)
    print("Success: discard(999) completed without error")
except KeyError as e:
    print(f"ERROR: KeyError raised: {e}")

# Test discarding an existing element (should work for both)
print("\n--- Testing discard of existing element ---")
os2 = OrderedSet([1, 2, 3])
print(f"OrderedSet before discard(2): {os2}")
os2.discard(2)
print(f"OrderedSet after discard(2): {os2}")