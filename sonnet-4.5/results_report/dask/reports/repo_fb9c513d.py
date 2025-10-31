import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

import dask.utils

# Test the specific failing value
n = 1125899906842624000
result = dask.utils.format_bytes(n)
print(f"format_bytes({n}) = {result!r}")
print(f"Length: {len(result)}")
print(f"Is n < 2**60? {n < 2**60} (2**60 = {2**60})")
print()

# Test additional examples to show the boundary
test_cases = [
    1124774006935781376,  # 999 PiB
    1125899906842624000,  # 1000 PiB
    1151795604700004352,  # 1023 PiB
]

print("Additional test cases:")
for test_n in test_cases:
    test_result = dask.utils.format_bytes(test_n)
    print(f"  format_bytes({test_n}) = {test_result!r} (length: {len(test_result)})")