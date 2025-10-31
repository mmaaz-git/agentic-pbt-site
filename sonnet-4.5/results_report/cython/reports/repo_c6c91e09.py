#!/usr/bin/env python3
"""Minimal reproduction of the validate_logit_bias bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.default_plugins.openai_models import SharedOptions

# Test case that crashes with TypeError
try:
    print("Testing logit_bias with {'1712': None}...")
    options = SharedOptions(logit_bias={"1712": None})
    print("Success - validated logit_bias:", options.logit_bias)
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Another test case from the bug report
try:
    print("\nTesting logit_bias with {'0': None, ':': None}...")
    options = SharedOptions(logit_bias={'0': None, ':': None})
    print("Success - validated logit_bias:", options.logit_bias)
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Test a valid case to show normal behavior
try:
    print("\nTesting valid logit_bias with {'1712': -50}...")
    options = SharedOptions(logit_bias={"1712": -50})
    print("Success - validated logit_bias:", options.logit_bias)
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")