#!/usr/bin/env python3
"""Better test script to reproduce the not_nulls bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.default_plugins.openai_models import not_nulls

# Test with empty dict (should work since no iteration happens)
print("Test with empty dict:")
empty_dict = {}
try:
    result = not_nulls(empty_dict)
    print(f"  Empty dict {{}} -> {result}")
except Exception as e:
    print(f"  ERROR: {e}")

print()

# Test with single-element dict
print("Test with single-element dict:")
single_dict = {"key": "value"}
try:
    result = not_nulls(single_dict)
    print(f"  {single_dict} -> {result}")
except Exception as e:
    print(f"  ERROR: {e}")

print()

# Test with multi-element dict
print("Test with multi-element dict:")
multi_dict = {"a": 1, "b": 2, "c": 3}
try:
    result = not_nulls(multi_dict)
    print(f"  {multi_dict} -> {result}")
except Exception as e:
    print(f"  ERROR: {e}")

print()

# Test with dict containing None values
print("Test with dict containing None values:")
none_dict = {"a": 1, "b": None, "c": 3}
try:
    result = not_nulls(none_dict)
    print(f"  {none_dict} -> {result}")
except Exception as e:
    print(f"  ERROR: {e}")

print()

# Test with dict.items() (correct usage)
print("Test with dict.items() (correct usage):")
none_dict = {"a": 1, "b": None, "c": 3}
try:
    result = not_nulls(none_dict.items())
    print(f"  {none_dict}.items() -> {result}")
except Exception as e:
    print(f"  ERROR: {e}")