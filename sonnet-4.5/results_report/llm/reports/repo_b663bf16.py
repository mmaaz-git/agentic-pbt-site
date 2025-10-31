#!/usr/bin/env python3
"""Minimal reproduction case for not_nulls bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.default_plugins.openai_models import not_nulls

# Test case with regular Python dict
data = {"temperature": 0.7, "max_tokens": None, "seed": 42}
print(f"Input dict: {data}")
print("Calling not_nulls(data)...")

try:
    result = not_nulls(data)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")