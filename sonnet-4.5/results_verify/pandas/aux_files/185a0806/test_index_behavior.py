import pandas as pd
import numpy as np

# Test Index behavior more carefully
print("=== Testing Index.take behavior ===\n")

# Test with integer Index
idx_int = pd.Index([10, 20, 30])
print(f"Original Index: {idx_int}")

print("\n1. Index.take with allow_fill=True, fill_value=None:")
result = idx_int.take([0, -1], allow_fill=True, fill_value=None)
print(f"   Result: {result}")
print(f"   Type: {type(result)}")
print(f"   Expected: Index([10, NaN]) but got Index([10, 30])")

print("\n2. Index.take with allow_fill=False:")
result = idx_int.take([0, -1], allow_fill=False)
print(f"   Result: {result}")
print(f"   Expected: Index([10, 30]) - correct!")

print("\n3. Index.take with allow_fill=True, fill_value=-999:")
try:
    result = idx_int.take([0, -1], allow_fill=True, fill_value=-999)
    print(f"   Result: {result}")
except ValueError as e:
    print(f"   ValueError: {e}")

# Test with float Index
print("\n=== Testing Float Index ===")
idx_float = pd.Index([10.0, 20.0, 30.0])
print(f"Original Index: {idx_float}")

print("\n1. Float Index.take with allow_fill=True, fill_value=None:")
result = idx_float.take([0, -1], allow_fill=True, fill_value=None)
print(f"   Result: {result}")
print(f"   Expected: Index([10.0, NaN])")

print("\n2. Float Index.take with allow_fill=True, fill_value=np.nan:")
result = idx_float.take([0, -1], allow_fill=True, fill_value=np.nan)
print(f"   Result: {result}")
print(f"   Expected: Index([10.0, NaN])")

# Test with string Index
print("\n=== Testing String Index ===")
idx_str = pd.Index(['a', 'b', 'c'])
print(f"Original Index: {idx_str}")

print("\n1. String Index.take with allow_fill=True, fill_value=None:")
result = idx_str.take([0, -1], allow_fill=True, fill_value=None)
print(f"   Result: {result}")
print(f"   Expected: Index(['a', None]) or Index(['a', NaN])")

print("\n2. String Index.take with allow_fill=True, fill_value='missing':")
try:
    result = idx_str.take([0, -1], allow_fill=True, fill_value='missing')
    print(f"   Result: {result}")
except ValueError as e:
    print(f"   ValueError: {e}")