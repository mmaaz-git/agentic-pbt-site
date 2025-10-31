#!/usr/bin/env python3
import sys
import os

# Activate the virtual environment's Python path
venv_path = "/home/npc/pbt/agentic-pbt/envs/llm_env"
sys.path.insert(0, os.path.join(venv_path, 'lib', 'python3.13', 'site-packages'))

import llm.utils

text = "Hello, World!"
max_length = 1
result = llm.utils.truncate_string(text, max_length=max_length)

print(f"Input: '{text}' (len={len(text)})")
print(f"max_length: {max_length}")
print(f"Output: '{result}' (len={len(result)})")
print(f"Constraint violated: {len(result)} > {max_length}")

# Test additional cases
print("\nAdditional test cases:")
for ml in [1, 2]:
    r = llm.utils.truncate_string(text, max_length=ml)
    print(f"  max_length={ml} → output length={len(r)} ('{r}')")