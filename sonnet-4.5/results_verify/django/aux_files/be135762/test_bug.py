#!/usr/bin/env python3
"""Test script to reproduce the django.template.Variable trailing dot bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.template import Variable

print("Testing Variable('2.')...")
try:
    var = Variable("2.")
    print(f"Success! Created Variable object")
    print(f"  literal: {var.literal}")
    print(f"  lookups: {var.lookups}")
    print(f"  var: {var.var}")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")

print("\nTesting Variable('1.')...")
try:
    var = Variable("1.")
    print(f"Success! Created Variable object")
    print(f"  literal: {var.literal}")
    print(f"  lookups: {var.lookups}")
    print(f"  var: {var.var}")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")

print("\nTesting Variable('10.')...")
try:
    var = Variable("10.")
    print(f"Success! Created Variable object")
    print(f"  literal: {var.literal}")
    print(f"  lookups: {var.lookups}")
    print(f"  var: {var.var}")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")

print("\nTesting Variable('2.0') for comparison...")
try:
    var = Variable("2.0")
    print(f"Success! Created Variable object")
    print(f"  literal: {var.literal}")
    print(f"  lookups: {var.lookups}")
    print(f"  var: {var.var}")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")