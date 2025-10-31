from django.db.backends.utils import truncate_name

# Test case from the bug report
identifier = 'SCHEMA"."VERYLONGTABLENAME'
length = 20

result = truncate_name(identifier, length=length)

print(f"Input: {identifier}")
print(f"Length limit: {length}")
print(f"Result: {result}")
print(f"Result length: {len(result.strip('\"'))}")
print()

# Check if the result exceeds the limit
stripped_result = result.strip('"')
if len(stripped_result) > length:
    print(f"ERROR: Result length ({len(stripped_result)}) exceeds limit ({length})")
    print(f"Excess characters: {len(stripped_result) - length}")
else:
    print(f"OK: Result length ({len(stripped_result)}) is within limit ({length})")

# Additional test cases
print("\n--- Additional Test Cases ---")

# Test 1: Without namespace - should truncate correctly
test1_id = "VERYLONGTABLENAME"
test1_result = truncate_name(test1_id, length=10)
print(f"\nTest 1 (no namespace):")
print(f"  Input: {test1_id}, Limit: 10")
print(f"  Result: {test1_result}")
print(f"  Result length: {len(test1_result.strip('\"'))}")

# Test 2: With namespace, very long table name
test2_id = 'SCHEMA"."VERYLONGTABLENAMETHATEXCEEDSLIMIT'
test2_result = truncate_name(test2_id, length=20)
print(f"\nTest 2 (namespace + very long table):")
print(f"  Input: {test2_id}, Limit: 20")
print(f"  Result: {test2_result}")
print(f"  Result length: {len(test2_result.strip('\"'))}")

# Test 3: Edge case with short namespace
test3_id = 'A"."BCDEFGHIJKLMNOPQRSTUVWXYZ'
test3_result = truncate_name(test3_id, length=15)
print(f"\nTest 3 (short namespace):")
print(f"  Input: {test3_id}, Limit: 15")
print(f"  Result: {test3_result}")
print(f"  Result length: {len(test3_result.strip('\"'))}")

# Assertion that should fail
print("\n--- Final Assertion ---")
try:
    assert len(result.strip('"')) <= length
    print("ASSERTION PASSED: Result is within length limit")
except AssertionError:
    print(f"ASSERTION FAILED: Result ({len(result.strip('\"'))}) exceeds limit ({length})")
    import traceback
    traceback.print_exc()