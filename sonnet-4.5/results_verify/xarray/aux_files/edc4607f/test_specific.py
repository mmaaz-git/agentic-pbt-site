import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from xarray.indexes import RangeIndex

# Test the specific case mentioned in the bug report
test_cases = [
    (0.0, 1.0, 1, True),   # The exact case from the bug report
    (0.0, 1.0, 1, False),  # Same but endpoint=False
    (5.0, 10.0, 1, True),  # Different values, same problem
    (5.0, 10.0, 1, False), # Different values, endpoint=False
]

for start, stop, num, endpoint in test_cases:
    print(f"\nTesting: start={start}, stop={stop}, num={num}, endpoint={endpoint}")
    try:
        index = RangeIndex.linspace(start, stop, num, endpoint=endpoint, dim="x")
        print(f"  Success: size={index.size}, start={index.start}, stop={index.stop}")
        # Let's also check what the actual value would be
        import numpy as np
        positions = np.arange(index.size)
        values = index.start + positions * index.step
        print(f"  Values: {values}")
    except ZeroDivisionError as e:
        print(f"  ✗ ZeroDivisionError: {e}")
        print(f"  Line 283 attempts: stop += (stop - start) / (num - 1)")
        print(f"  With num=1: stop += ({stop} - {start}) / (1 - 1) = division by zero")