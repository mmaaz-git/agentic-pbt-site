import numpy as np
import pandas as pd
from pandas.api.extensions import take

# Test 1: Basic test with Index vs Array
print("Test 1: Basic Index vs Array with fill_value=None")
print("-" * 50)
index = pd.Index([10.0, 20.0, 30.0])
arr = np.array([10.0, 20.0, 30.0])

index_result = take(index, [0, -1, 2], allow_fill=True, fill_value=None)
array_result = take(arr, [0, -1, 2], allow_fill=True, fill_value=None)

print(f"Index result: {list(index_result)}")
print(f"Array result: {list(array_result)}")
print(f"Inconsistency: Index[-1] = {index_result[1]}, Array[-1] = {array_result[1]}")
print()

# Test 2: With explicit fill_value
print("Test 2: Index vs Array with explicit fill_value=999")
print("-" * 50)
index_result2 = take(index, [0, -1, 2], allow_fill=True, fill_value=999)
array_result2 = take(arr, [0, -1, 2], allow_fill=True, fill_value=999)

print(f"Index result: {list(index_result2)}")
print(f"Array result: {list(array_result2)}")
print(f"Both consistent: Index[-1] = {index_result2[1]}, Array[-1] = {array_result2[1]}")
print()

# Test 3: Without allow_fill
print("Test 3: Index vs Array with allow_fill=False")
print("-" * 50)
index_result3 = take(index, [0, -1, 2], allow_fill=False)
array_result3 = take(arr, [0, -1, 2], allow_fill=False)

print(f"Index result: {list(index_result3)}")
print(f"Array result: {list(array_result3)}")
print(f"Both consistent: Index[-1] = {index_result3[1]}, Array[-1] = {array_result3[1]}")
print()

# Test 4: Check documentation example
print("Test 4: Documentation example")
print("-" * 50)
doc_example = take(np.array([10, 20, 30]), [0, 0, -1], allow_fill=True)
print(f"Doc example result: {list(doc_example)}")
print("Documentation says this should be [10., 10., nan]")
print(f"Result matches documentation: {pd.isna(doc_example[2])}")
print()

# Test 5: Check with pd.Series (which has same issue as Index)
print("Test 5: Series behavior")
print("-" * 50)
series = pd.Series([10.0, 20.0, 30.0])
series_result = take(series, [0, -1, 2], allow_fill=True, fill_value=None)
print(f"Series result: {list(series_result)}")
print(f"Series has same issue: Series[-1] = {series_result[1]}")