#!/usr/bin/env python3
"""Test to understand tokenization of default statements more carefully"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

# Test what actually happens
test_str = "default "
print(f"Original: {repr(test_str)}")
print(f"After strip: {repr(test_str.strip())}")
print(f"'{test_str.strip()}'.startswith('default '): {test_str.strip().startswith('default ')}")

# Now let's trace the actual parse_expr logic
expr = "default "
expr_stripped = expr.strip()
print(f"\nexpr = {repr(expr)}")
print(f"expr.strip() = {repr(expr_stripped)}")
print(f"expr.startswith('default ') before strip = {expr.startswith('default ')}")

# After stripping on line 751
print(f"\nAfter line 751 (expr = expr.strip()):")
expr = expr.strip()
print(f"expr = {repr(expr)}")
print(f"expr.startswith('default ') = {expr.startswith('default ')}")

# So it should not enter parse_default