#!/usr/bin/env python3
"""Minimal reproduction of truncate_string bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.utils import truncate_string

# Test cases that violate the max_length contract
test_cases = [
    ('00', 1),
    ('000', 2),
    ('0000', 3),
]

print("Demonstrating truncate_string violating max_length contract")
print("=" * 60)

for text, max_length in test_cases:
    result = truncate_string(text, max_length=max_length)
    print(f"\nInput: text='{text}', max_length={max_length}")
    print(f"Result: '{result}'")
    print(f"Result length: {len(result)}")
    print(f"Expected max length: {max_length}")
    print(f"Contract violated: {len(result) > max_length}")