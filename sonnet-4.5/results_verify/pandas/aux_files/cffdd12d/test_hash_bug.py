from hypothesis import given, strategies as st, assume
import numpy as np
import pandas.util


@given(st.text(min_size=1, max_size=15))
def test_hash_array_validates_hash_key_length(hash_key):
    assume(len(hash_key.encode('utf8')) < 16)
    arr = np.array(['a', 'b', 'c'], dtype=object)

    try:
        result = pandas.util.hash_array(arr, hash_key=hash_key)
        print(f"Test PASSED for hash_key='{hash_key}' (len={len(hash_key.encode('utf8'))})")
    except ValueError as e:
        print(f"Test FAILED with ValueError for hash_key='{hash_key}' (len={len(hash_key.encode('utf8'))}): {e}")
        raise  # Re-raise to indicate test failure

# Run the test
print("Running hypothesis test...")
try:
    test_hash_array_validates_hash_key_length()
except Exception as e:
    print(f"Test confirmed the bug exists: {e}")