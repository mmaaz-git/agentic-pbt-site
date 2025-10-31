import numpy as np
import pandas as pd
import pandas.arrays as pa

def test_array_type(name, arr):
    print(f"\nTesting {name}:")
    print(f"Array: {arr}")

    codes, uniques = arr.factorize()
    print(f"Codes: {codes}")
    print(f"Uniques: {uniques}")
    print(f"Uniques length: {len(uniques)}")

    try:
        reconstructed = uniques[codes]
        print(f"Reconstruction successful: {reconstructed}")
    except IndexError as e:
        print(f"IndexError: {e}")

    # Test with use_na_sentinel=False
    codes2, uniques2 = arr.factorize(use_na_sentinel=False)
    print(f"With use_na_sentinel=False - Codes: {codes2}, Uniques: {uniques2}")
    try:
        reconstructed2 = uniques2[codes2]
        print(f"Reconstruction successful with sentinel=False")
    except IndexError as e:
        print(f"IndexError with sentinel=False: {e}")

# Test BooleanArray
print("="*60)
bool_arr = pa.BooleanArray(np.array([True], dtype=bool), mask=np.array([True]))
test_array_type("BooleanArray", bool_arr)

# Test FloatingArray
print("="*60)
float_arr = pa.FloatingArray(np.array([1.0], dtype='float64'), mask=np.array([True]))
test_array_type("FloatingArray", float_arr)

# Test ArrowStringArray
print("="*60)
try:
    import pyarrow
    # Create ArrowStringArray with NA value
    arrow_str_arr = pa.ArrowStringArray._from_sequence([None])
    test_array_type("ArrowStringArray", arrow_str_arr)
except (ImportError, Exception) as e:
    print(f"ArrowStringArray test skipped: {e}")

# Also test ArrowExtensionArray if available
print("="*60)
try:
    import pyarrow
    # Create an ArrowExtensionArray with all NA values
    pa_array = pyarrow.array([None], type=pyarrow.int64())
    arrow_ext_arr = pd.arrays.ArrowExtensionArray(pa_array)
    print("\nTesting ArrowExtensionArray:")
    print(f"Array: {arrow_ext_arr}")

    codes, uniques = arrow_ext_arr.factorize()
    print(f"Codes: {codes}")
    print(f"Uniques: {uniques}")
    print(f"Uniques length: {len(uniques)}")

    try:
        reconstructed = uniques[codes]
        print(f"Reconstruction successful: {reconstructed}")
    except IndexError as e:
        print(f"IndexError: {e}")

except (ImportError, AttributeError) as e:
    print(f"ArrowExtensionArray test skipped: {e}")