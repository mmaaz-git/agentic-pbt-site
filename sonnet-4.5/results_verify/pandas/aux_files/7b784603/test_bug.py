import numpy as np
import pandas as pd

print("Testing hash_array with different hash_key values...")
print("=" * 60)

# Test with numeric array
arr_numeric = np.array([1, 2, 3], dtype=np.int64)

# Test with various hash_key lengths for numeric array
test_keys = [
    "short",
    "0123456789123456",  # 16 bytes - default
    "this_is_a_longer_key_than_16_bytes"
]

print("\n1. Testing with NUMERIC array (int64):")
for key in test_keys:
    try:
        result = pd.util.hash_array(arr_numeric, hash_key=key)
        print(f"  Key: '{key}' (len={len(key)}) - SUCCESS")
        print(f"    Result: {result}")
    except ValueError as e:
        print(f"  Key: '{key}' (len={len(key)}) - ERROR: {e}")
    except Exception as e:
        print(f"  Key: '{key}' (len={len(key)}) - UNEXPECTED ERROR: {e}")

print("\n2. Testing with OBJECT array (strings):")
arr_object = np.array(['a', 'b', 'c'], dtype=object)

for key in test_keys:
    try:
        result = pd.util.hash_array(arr_object, hash_key=key)
        print(f"  Key: '{key}' (len={len(key)}) - SUCCESS")
        print(f"    Result: {result}")
    except ValueError as e:
        print(f"  Key: '{key}' (len={len(key)}) - ERROR: {e}")
    except Exception as e:
        print(f"  Key: '{key}' (len={len(key)}) - UNEXPECTED ERROR: {e}")

print("\n3. Testing hash consistency with different keys:")
print("-" * 40)

# Test if different hash_keys produce different hashes for numeric arrays
arr = np.array([1, 2, 3, 4, 5], dtype=np.int64)
key1 = "0123456789123456"
key2 = "abcdefghijklmnop"

hash1 = pd.util.hash_array(arr, hash_key=key1)
hash2 = pd.util.hash_array(arr, hash_key=key2)

print(f"Numeric array with key1: {hash1}")
print(f"Numeric array with key2: {hash2}")
print(f"Hashes are different: {not np.array_equal(hash1, hash2)}")

print("\n4. Testing the property-based test from the bug report:")
print("-" * 40)

from hypothesis import given, strategies as st, settings, assume
import pandas.util

@given(
    st.lists(st.integers(min_value=-2**63, max_value=2**63-1), min_size=1),
    st.text(min_size=16, max_size=16, alphabet=st.characters()),
    st.text(min_size=16, max_size=16, alphabet=st.characters())
)
@settings(max_examples=10)
def test_hash_array_different_keys_produce_different_hashes(values, key1, key2):
    assume(key1 != key2)
    arr = np.array(values, dtype=np.int64)
    hash1 = pandas.util.hash_array(arr, hash_key=key1)
    hash2 = pandas.util.hash_array(arr, hash_key=key2)

    print(f"  Testing with {len(values)} values")
    print(f"    Key1: {repr(key1[:5])}... Key2: {repr(key2[:5])}...")
    print(f"    Hashes equal? {np.array_equal(hash1, hash2)}")

    if len(arr) > 1:
        assert not np.array_equal(hash1, hash2), f"Hashes should differ but are the same!"

print("Running property-based test...")
try:
    test_hash_array_different_keys_produce_different_hashes()
    print("Property test PASSED")
except AssertionError as e:
    print(f"Property test FAILED: {e}")