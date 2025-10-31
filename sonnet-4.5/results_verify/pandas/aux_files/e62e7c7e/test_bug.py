import numpy as np
import pandas.core.algorithms as algorithms

# Test 1: Basic reproduction of the bug
print("Test 1: Basic reproduction with big-endian array")
try:
    arr_big_endian = np.array([1, 2, 3, 2, 1], dtype='>i8')
    print(f"Input array: {arr_big_endian}, dtype: {arr_big_endian.dtype}")
    codes, uniques = algorithms.factorize(arr_big_endian)
    print(f"Success! Codes: {codes}, Uniques: {uniques}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\n" + "="*50 + "\n")

# Test 2: Control test with native byte order
print("Test 2: Control test with native byte order")
try:
    arr_native = np.array([1, 2, 3, 2, 1], dtype='i8')  # Native byte order
    print(f"Input array: {arr_native}, dtype: {arr_native.dtype}")
    codes, uniques = algorithms.factorize(arr_native)
    print(f"Success! Codes: {codes}, Uniques: {uniques}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\n" + "="*50 + "\n")

# Test 3: Test with different big-endian dtypes
print("Test 3: Testing various big-endian dtypes")
test_dtypes = ['>i8', '>i4', '>i2', '>f8', '>f4']
for dtype in test_dtypes:
    try:
        arr = np.array([1, 2, 3], dtype=dtype)
        print(f"Testing dtype {dtype}: ", end="")
        codes, uniques = algorithms.factorize(arr)
        print(f"Success!")
    except Exception as e:
        print(f"Failed - {type(e).__name__}")

print("\n" + "="*50 + "\n")

# Test 4: Check if pandas Series/DataFrame handle big-endian
import pandas as pd
print("Test 4: pandas Series/DataFrame with big-endian arrays")
try:
    arr_big = np.array([1, 2, 3], dtype='>i8')
    series = pd.Series(arr_big)
    print(f"Series created successfully with dtype: {series.dtype}")
    print(series)
except Exception as e:
    print(f"Series failed: {e}")

try:
    df = pd.DataFrame({'col': arr_big})
    print(f"\nDataFrame created successfully with dtype: {df['col'].dtype}")
    print(df)
except Exception as e:
    print(f"DataFrame failed: {e}")