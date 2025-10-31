import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from xarray.core.formatting_html import _load_static_files

result1 = _load_static_files()
print(f"First call: {result1[0][:50]}...")
print(f"Type of result: {type(result1)}")
print(f"Number of items: {len(result1)}")

original_value = result1[0]

result1[0] = "MUTATED_CONTENT"

result2 = _load_static_files()
print(f"\nSecond call: {result2[0]}")

print(f"\nDoes cache return mutated content? {result2[0] == 'MUTATED_CONTENT'}")
print(f"Cache corrupted: {result2[0] != original_value}")

assert result2[0] == "MUTATED_CONTENT", "This assertion should pass if the bug exists"
print("\nAssertion passed - bug is confirmed!")