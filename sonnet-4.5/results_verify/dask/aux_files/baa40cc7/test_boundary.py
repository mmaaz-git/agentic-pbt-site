import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from dask.utils import format_bytes

# Test the boundary cases
print("Testing boundary cases for format_bytes:")
print("=" * 50)

# Test around the 1000 PiB boundary
for multiplier in [999, 999.9, 999.99, 1000, 1000.01, 1001, 1023, 1024]:
    n = int(2**50 * multiplier)
    if n < 2**60:  # Only test values under the documented limit
        result = format_bytes(n)
        print(f"{multiplier:7.2f} PiB: format_bytes({n:20}) = '{result:11}' (len={len(result):2})")

print("\n" + "=" * 50)
print("Testing the maximum documented value:")

# Test the maximum value mentioned in the docstring
max_val = 2**60 - 1
result = format_bytes(max_val)
print(f"format_bytes(2^60 - 1) = '{result}' (length: {len(result)})")
print(f"2^60 - 1 = {max_val}")
print(f"This is approximately {max_val / 2**50:.2f} PiB")