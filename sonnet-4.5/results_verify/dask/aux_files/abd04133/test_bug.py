#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from dask.diagnostics.profile_visualize import unquote

print("Testing basic failing case...")
try:
    expr = (set, [(list, [])])
    result = unquote(expr)
    print(f"Result: {result}")
except TypeError as e:
    print(f"TypeError caught: {e}")

print("\nTesting additional failing cases...")
test_cases = [
    ((set, [[1, 2], [3, 4]]), "set with plain lists"),
    ((set, [(dict, [['a', 1]])]), "set with dict constructor"),
    ((set, [(set, [1, 2])]), "set with nested set constructor"),
]

for expr, description in test_cases:
    print(f"\nTesting {description}: {expr}")
    try:
        result = unquote(expr)
        print(f"  Result: {result}")
    except TypeError as e:
        print(f"  TypeError caught: {e}")

print("\nTesting cases that should work...")
working_cases = [
    ((list, [(list, [])]), "list with nested list"),
    ((tuple, [(list, [])]), "tuple with nested list"),
    ((set, [1, 2, 3]), "set with hashable elements"),
    ((dict, [[['a', 1], ['b', 2]]]), "dict constructor"),
]

for expr, description in working_cases:
    print(f"\nTesting {description}: {expr}")
    try:
        result = unquote(expr)
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Unexpected error: {e}")