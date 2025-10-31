import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env')

from dask.utils import format_bytes

# Test values around 1000 PiB
test_cases = [
    999 * 2**50,     # 999 PiB
    999.9 * 2**50,   # ~999.9 PiB
    1000 * 2**50,    # 1000 PiB
    1001 * 2**50,    # 1001 PiB
    1023 * 2**50,    # 1023 PiB (max before EiB would kick in if it existed)
]

print("Testing edge cases around 1000 PiB:")
print("-" * 50)

for n in test_cases:
    result = format_bytes(int(n))
    print(f"n = {int(n):22d}")
    print(f"  ≈ {n/2**50:7.2f} PiB")
    print(f"  format_bytes() = '{result}'")
    print(f"  Length = {len(result)} chars")
    if len(result) > 10:
        print(f"  ⚠️  EXCEEDS 10 char limit!")
    print()

# Also test the boundary where the function switches to PiB
print("\nTesting boundaries for unit transitions:")
print("-" * 50)

# Test the 0.9 threshold
for unit, size in [("Pi", 2**50), ("Ti", 2**40), ("Gi", 2**30)]:
    threshold = int(size * 0.9)
    just_below = threshold - 1
    at_threshold = threshold

    print(f"\n{unit}B transition (at {size * 0.9:.0f} bytes):")
    print(f"  Just below: {format_bytes(just_below)} (len={len(format_bytes(just_below))})")
    print(f"  At threshold: {format_bytes(at_threshold)} (len={len(format_bytes(at_threshold))})")

# Test all values that will produce >= 1000 in any unit
print("\n\nChecking all potential 11+ char outputs:")
print("-" * 50)

for prefix, k in [("Pi", 2**50), ("Ti", 2**40), ("Gi", 2**30), ("Mi", 2**20), ("ki", 2**10)]:
    # Values that will produce >= 1000.00 of this unit
    test_val = int(1000 * k)
    if test_val < 2**60:
        result = format_bytes(test_val)
        print(f"{prefix}B: format_bytes({test_val}) = '{result}' ({len(result)} chars)")
        if len(result) > 10:
            print(f"  ⚠️  EXCEEDS 10 char limit!")