import sys

# Test case 1: Simple dict with underscores
print("Test 1: Dict with underscore headers")
headers = {'X_Header': 'v1', 'Y_Header': 'v2', 'Z_Header': 'v3'}
print(f"Before: {list(headers.keys())}")
try:
    for k in headers:
        if "_" in k:
            del headers[k]
    print("No error")
except RuntimeError as e:
    print(f"RuntimeError: {e}")
print(f"After: {list(headers.keys())}")
print(f"Remaining underscore headers: {[k for k in headers.keys() if '_' in k]}")

# Test case 2: Mixed headers
print("\nTest 2: Mixed headers")
headers2 = {'X_Header': 'v1', 'Normal-Header': 'v2', 'Y_Header': 'v3'}
print(f"Before: {list(headers2.keys())}")
try:
    for k in headers2:
        if "_" in k:
            del headers2[k]
    print("No error")
except RuntimeError as e:
    print(f"RuntimeError: {e}")
print(f"After: {list(headers2.keys())}")
print(f"Remaining underscore headers: {[k for k in headers2.keys() if '_' in k]}")