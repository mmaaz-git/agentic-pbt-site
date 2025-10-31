#!/usr/bin/env python
"""Minimal reproduction of the django.template.Variable trailing dot bug"""

from django.template import Variable, Context

# Create a Variable with a numeric string ending in a dot
var = Variable('42.')

# Show the inconsistent internal state
print(f"Variable created with '42.':")
print(f"  literal: {var.literal}")
print(f"  lookups: {var.lookups}")
print()

# Try to resolve it
ctx = Context({})
try:
    result = var.resolve(ctx)
    print(f"Resolution succeeded: {result}")
except Exception as e:
    print(f"Resolution failed with: {type(e).__name__}: {e}")