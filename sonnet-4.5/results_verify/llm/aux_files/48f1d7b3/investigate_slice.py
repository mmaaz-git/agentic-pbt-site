#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

# Let's understand how Python slicing works with negative indices
text = "Hello world"
print(f"Original text: '{text}' (length={len(text)})")
print()

for max_length in range(0, 5):
    # This is what the code does: text[: max_length - 3] + "..."
    slice_end = max_length - 3
    truncated = text[:slice_end]
    result = truncated + "..."
    print(f"max_length={max_length}:")
    print(f"  slice_end = max_length - 3 = {max_length} - 3 = {slice_end}")
    print(f"  text[:{slice_end}] = '{truncated}'")
    print(f"  result = '{result}' (length={len(result)})")
    print()

print("Python slicing behavior with negative indices:")
print(f"text[:-3] = '{text[:-3]}' (excludes last 3 chars)")
print(f"text[:-2] = '{text[:-2]}' (excludes last 2 chars)")
print(f"text[:-1] = '{text[:-1]}' (excludes last 1 char)")
print(f"text[:0]  = '{text[:0]}' (empty string)")