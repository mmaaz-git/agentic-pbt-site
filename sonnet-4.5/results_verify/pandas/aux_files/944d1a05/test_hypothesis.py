from hypothesis import given, strategies as st
import numpy as np
from pandas.core.util.hashing import hash_array

@given(st.lists(st.text(max_size=10), min_size=1),
       st.text(min_size=16, max_size=16))
def test_hash_array_object_different_hash_keys(str_list, hash_key1):
    arr = np.array(str_list, dtype=object)
    hash_key2 = hash_key1[:15] + ('x' if hash_key1[15] != 'x' else 'y')

    result1 = hash_array(arr, hash_key=hash_key1)
    result2 = hash_array(arr, hash_key=hash_key2)

    assert not np.array_equal(result1, result2)

# Run a simple test with the specific failing input
str_list = ['']
hash_key1 = '000000000000000\x80'

print(f"Testing with str_list={str_list}, hash_key1={repr(hash_key1)}")
print(f"hash_key1 length: {len(hash_key1)} characters")
print(f"hash_key1 encoded length: {len(hash_key1.encode('utf8'))} bytes")

try:
    test_hash_array_object_different_hash_keys(str_list, hash_key1)
    print("Test passed")
except Exception as e:
    print(f"Test failed with: {type(e).__name__}: {e}")