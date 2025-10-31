#!/usr/bin/env python3
"""Minimal reproduction of validate_logit_bias exception handling bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.default_plugins.openai_models import SharedOptions

# Test case 1: Value out of range (too high)
options = SharedOptions()
try:
    result = options.validate_logit_bias({"100": 150})
    print(f"Test 1 passed unexpectedly: {result}")
except ValueError as e:
    print(f"Test 1 - Out of range high (150): {e}")

# Test case 2: Value out of range (too low)
try:
    result = options.validate_logit_bias({"100": -150})
    print(f"Test 2 passed unexpectedly: {result}")
except ValueError as e:
    print(f"Test 2 - Out of range low (-150): {e}")

# Test case 3: Invalid key (cannot convert to int)
try:
    result = options.validate_logit_bias({"abc": 50})
    print(f"Test 3 passed unexpectedly: {result}")
except ValueError as e:
    print(f"Test 3 - Invalid key ('abc'): {e}")

# Test case 4: Invalid value (cannot convert to int)
try:
    result = options.validate_logit_bias({"100": "not_a_number"})
    print(f"Test 4 passed unexpectedly: {result}")
except ValueError as e:
    print(f"Test 4 - Invalid value ('not_a_number'): {e}")

# Test case 5: Valid input (should pass)
try:
    result = options.validate_logit_bias({"100": 50})
    print(f"Test 5 - Valid input: {result}")
except ValueError as e:
    print(f"Test 5 failed unexpectedly: {e}")