#!/usr/bin/env python3
"""Test what happens with -O flag"""

import numpy.f2py.symbolic as symbolic

print("Testing with -O flag (assertions disabled)")
try:
    result = symbolic.eliminate_quotes('"')
    print(f"Result: {result}")
    print("No error raised! The quote remained in the output.")
except AssertionError:
    print("AssertionError still raised (assertions enabled)")
except Exception as e:
    print(f"Other exception: {type(e).__name__}: {e}")