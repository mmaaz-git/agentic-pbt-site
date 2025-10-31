import numpy as np
import pandas as pd
from pandas.core.util.hashing import hash_array, hash_pandas_object

print("=" * 50)
print("Testing hash_array with negative integers")
print("=" * 50)

arr_int32 = np.array([-1], dtype=np.int32)
arr_int64 = np.array([-1], dtype=np.int64)

hash_int32 = hash_array(arr_int32)[0]
hash_int64 = hash_array(arr_int64)[0]

print(f"Hash of -1 as int32: {hash_int32}")
print(f"Hash of -1 as int64: {hash_int64}")
print(f"Equal: {hash_int32 == hash_int64}")

print("\n" + "=" * 50)
print("Testing hash_pandas_object with Series")
print("=" * 50)

s_int32 = pd.Series([-1, -2, -10], dtype='int32')
s_int64 = pd.Series([-1, -2, -10], dtype='int64')

hash_s32 = hash_pandas_object(s_int32, index=False)
hash_s64 = hash_pandas_object(s_int64, index=False)

print(f"Series int32 hashes: {list(hash_s32)}")
print(f"Series int64 hashes: {list(hash_s64)}")
print(f"Series hashes equal: {hash_s32.equals(hash_s64)}")

print("\n" + "=" * 50)
print("Testing with positive values for comparison")
print("=" * 50)

arr_pos_int32 = np.array([1], dtype=np.int32)
arr_pos_int64 = np.array([1], dtype=np.int64)

hash_pos_int32 = hash_array(arr_pos_int32)[0]
hash_pos_int64 = hash_array(arr_pos_int64)[0]

print(f"Hash of 1 as int32: {hash_pos_int32}")
print(f"Hash of 1 as int64: {hash_pos_int64}")
print(f"Positive values work correctly:")
print(f"Equal: {hash_pos_int32 == hash_pos_int64}")

print("\n" + "=" * 50)
print("Testing more negative values")
print("=" * 50)

for value in [-1, -2, -10, -100]:
    arr_int32 = np.array([value], dtype=np.int32)
    arr_int64 = np.array([value], dtype=np.int64)

    hash_int32 = hash_array(arr_int32)[0]
    hash_int64 = hash_array(arr_int64)[0]

    print(f"Value {value}: int32={hash_int32}, int64={hash_int64}, equal={hash_int32 == hash_int64}")

print("\n" + "=" * 50)
print("Testing more positive values")
print("=" * 50)

for value in [1, 2, 10, 100]:
    arr_int32 = np.array([value], dtype=np.int32)
    arr_int64 = np.array([value], dtype=np.int64)

    hash_int32 = hash_array(arr_int32)[0]
    hash_int64 = hash_array(arr_int64)[0]

    print(f"Value {value}: int32={hash_int32}, int64={hash_int64}, equal={hash_int32 == hash_int64}")