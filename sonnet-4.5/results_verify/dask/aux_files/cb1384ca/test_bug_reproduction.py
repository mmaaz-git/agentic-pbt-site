#!/usr/bin/env python3

# Test 1: Reproduce the exact bug
print("Test 1: Reproducing the bug with (dict, [])")
try:
    from dask.diagnostics.profile_visualize import unquote
    expr = (dict, [])
    result = unquote(expr)
    print(f"  Result: {result}")
    print("  Bug NOT reproduced - function did not crash")
except IndexError as e:
    print(f"  IndexError occurred as expected: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"  Unexpected error: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 2: Check istask function behavior
print("Test 2: Checking if (dict, []) is considered a task")
from dask.core import istask
expr = (dict, [])
is_task = istask(expr)
print(f"  istask((dict, [])): {is_task}")
print(f"  callable(dict): {callable(dict)}")

print()

# Test 3: Test the normal case with dict
print("Test 3: Testing normal case with dict task")
try:
    expr = (dict, [[('a', 1), ('b', 2)]])
    result = unquote(expr)
    print(f"  Input: {expr}")
    print(f"  Result: {result}")
except Exception as e:
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 4: Test with other constructors
print("Test 4: Testing with other constructors")
for constructor in [tuple, list, set]:
    try:
        expr = (constructor, [])
        result = unquote(expr)
        print(f"  unquote(({constructor.__name__}, [])): {result}")
    except Exception as e:
        print(f"  Error with {constructor.__name__}: {e}")