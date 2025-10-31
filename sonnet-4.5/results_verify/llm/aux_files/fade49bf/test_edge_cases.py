#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.utils import schema_dsl

test_cases = [
    "field1 str, , field2 int",  # Original test case
    ", , ,",  # Only commas
    "   ,   ,   ",  # Whitespace and commas
    "",  # Empty string
    "     ",  # Only whitespace
    ",field1",  # Leading comma
    "field1,",  # Trailing comma
    "field1,,field2",  # Double comma
    "\n\n\n",  # Only newlines
    "field1\n\nfield2",  # Double newline
]

for i, test in enumerate(test_cases):
    print(f"\nTest {i}: {repr(test)}")
    try:
        result = schema_dsl(test)
        print(f"  Success: {result}")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")