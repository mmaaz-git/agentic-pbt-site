import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from dask.utils import format_bytes

# Test the specific case mentioned
n = 1000 * 2**50
result = format_bytes(n)

print("=== Testing specific case ===")
print(f"Value: {n}")
print(f"Output: '{result}'")
print(f"Length: {len(result)}")
print(f"2**60: {2**60}")
print(f"Value < 2**60: {n < 2**60}")
print(f"Length <= 10: {len(result) <= 10}")

# Test more edge cases
print("\n=== Testing edge cases near 2**60 ===")
test_values = [
    (999 * 2**50, "999 PiB"),
    (1000 * 2**50, "1000 PiB"),
    (1023 * 2**50, "1023 PiB"),
    (1024 * 2**50 - 1, "max before 2**60"),
    (2**60 - 1, "2**60 - 1"),
    (2**60, "2**60 exactly")
]

for val, desc in test_values:
    if val < 2**61:  # Safety check to avoid huge numbers
        res = format_bytes(val)
        print(f"{desc:20} -> '{res:12}' (len={len(res):2}) < 2**60: {val < 2**60}")

# Run the hypothesis test too
print("\n=== Running Hypothesis Test ===")
from hypothesis import given, strategies as st, settings

@settings(max_examples=1000)
@given(st.integers(min_value=0, max_value=2**60-1))
def test_format_bytes_length_constraint_documented(n):
    """Property: format_bytes should output <= 10 chars for n < 2**60 (documented claim)"""
    result = format_bytes(n)

    if n < 2**60:
        if len(result) > 10:
            print(f"VIOLATION: format_bytes({n}) = '{result}' (length={len(result)} > 10)")
            return False
    return True

# Try to find violations
try:
    test_format_bytes_length_constraint_documented()
    print("No violations found in hypothesis test")
except AssertionError as e:
    print(f"Hypothesis test found violations")
except Exception as e:
    print(f"Error running hypothesis test: {e}")

# Manually verify the calculation for 2**60
print("\n=== Verifying 2**60 calculation ===")
print(f"2**50 = {2**50}")
print(f"1024 * 2**50 = {1024 * 2**50}")
print(f"2**60 = {2**60}")
print(f"They are equal: {1024 * 2**50 == 2**60}")