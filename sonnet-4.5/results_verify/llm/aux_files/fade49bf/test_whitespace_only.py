#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.utils import schema_dsl

# Try to create a case where field_parts would be empty
# This would require a field that passes the initial filter but has no content after split()

# After the filter on line 382, we would need a field that:
# 1. Is not empty after strip() (passes the filter)
# 2. But when split() is called on line 395, produces an empty list

# This is impossible because if field.strip() is not empty,
# field_info.strip().split() will always have at least one element

# Let's verify this logic:
test_strings = [
    "  ",  # This gets filtered out at line 382
    "\t",  # This gets filtered out at line 382
    "\n",  # This gets filtered out at line 382
]

for test in test_strings:
    # Simulate the filter
    if test.strip():
        print(f"'{test}' passes filter")
        parts = test.strip().split()
        print(f"  Parts: {parts}")
        if not parts:
            print("  ERROR: Would cause IndexError!")
    else:
        print(f"'{test}' filtered out (empty after strip)")