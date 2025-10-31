import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from dask.utils import format_bytes

# Find the exact boundary where length exceeds 10
print("=== Finding exact boundary where output exceeds 10 characters ===")

# Test around 1000 PiB
base = 2**50
for multiplier in [999.5, 999.9, 999.99, 1000, 1000.01]:
    n = int(multiplier * base)
    result = format_bytes(n)
    print(f"{multiplier:7.2f} PiB: n={n:19} -> '{result}' (len={len(result)})")

print("\n=== Understanding the threshold calculation ===")
# The code uses: if n >= k * 0.9
# For PiB: k = 2**50
# So threshold is 2**50 * 0.9

threshold = 2**50 * 0.9
print(f"PiB threshold: {threshold:.0f}")
print(f"That's {threshold / 2**50:.6f} PiB")

# When does format give 4 digits before decimal?
print("\n=== When do we get 4 digits? ===")
for n_pib in [999.994, 999.995, 999.996, 1000, 1023, 1024]:
    n = n_pib * 2**50
    if n < 2**61:  # Safety
        result = format_bytes(n)
        print(f"{n_pib:7.3f} PiB -> '{result}' (len={len(result)})")

# What about values just below 2**60?
print("\n=== Values near 2**60 ===")
for offset in [-100, -10, -1, 0]:
    n = 2**60 + offset
    if n > 0:
        result = format_bytes(n)
        in_range = "YES" if n < 2**60 else "NO"
        print(f"2**60 {offset:+4}: {n} -> '{result}' (len={len(result)}) in range: {in_range}")