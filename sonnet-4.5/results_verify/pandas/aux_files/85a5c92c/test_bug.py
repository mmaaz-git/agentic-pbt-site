#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from pandas.api.types import is_re_compilable
import re

print("Testing is_re_compilable with invalid regex patterns:")
print("-" * 50)

invalid_patterns = ['?', '*', '+', '\\', '[', '(', ')']

for pattern in invalid_patterns:
    try:
        result = is_re_compilable(pattern)
        print(f"'{pattern}': {result}")
    except Exception as e:
        print(f"'{pattern}': CRASH - {type(e).__name__}: {e}")

print("\n" + "-" * 50)
print("\nTesting what happens with re.compile directly on these patterns:")
print("-" * 50)

for pattern in invalid_patterns:
    try:
        re.compile(pattern)
        print(f"'{pattern}': Successfully compiled")
    except re.error as e:
        print(f"'{pattern}': re.error - {e}")
    except Exception as e:
        print(f"'{pattern}': {type(e).__name__}: {e}")

print("\n" + "-" * 50)
print("\nTesting valid patterns with is_re_compilable:")
print("-" * 50)

valid_patterns = ['.*', 'abc', '[a-z]+', r'\d+', '(hello|world)']

for pattern in valid_patterns:
    try:
        result = is_re_compilable(pattern)
        print(f"'{pattern}': {result}")
    except Exception as e:
        print(f"'{pattern}': CRASH - {type(e).__name__}: {e}")