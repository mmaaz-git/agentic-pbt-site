import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from dask.utils import format_bytes

n = 1_125_894_277_343_089_729
result = format_bytes(n)

print(f"format_bytes({n}) = '{result}'")
print(f"Length: {len(result)} characters")
print(f"Expected: <= 10 characters")
print(f"Violates documented invariant: {len(result) > 10}")

# Test a few more cases to understand the pattern
test_cases = [
    2**50 * 999,  # Just under 1000 PiB
    2**50 * 1000,  # Exactly 1000 PiB
    2**50 * 1001,  # Just over 1000 PiB
    2**60 - 1,  # Maximum value according to the docstring
]

print("\nAdditional test cases:")
for test_n in test_cases:
    test_result = format_bytes(test_n)
    print(f"format_bytes({test_n}) = '{test_result}' (length: {len(test_result)})")