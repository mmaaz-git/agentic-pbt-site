#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

print("Test 1: Reproducing the exact bug")
try:
    from Cython.Build.Dependencies import resolve_depend
    result = resolve_depend("", ())
    print(f"Result: {result}")
except IndexError as e:
    print(f"IndexError caught: {e}")
except Exception as e:
    print(f"Other exception: {e}")

print("\nTest 2: Testing with empty string and non-empty include_dirs")
try:
    result = resolve_depend("", ("/usr/include", "/usr/local/include"))
    print(f"Result: {result}")
except IndexError as e:
    print(f"IndexError caught: {e}")

print("\nTest 3: Testing with valid inputs")
try:
    result = resolve_depend("test.h", ())
    print(f"Result for 'test.h' with empty dirs: {result}")
except Exception as e:
    print(f"Exception: {e}")

print("\nTest 4: Testing angle bracket syntax with non-empty string")
try:
    result = resolve_depend("<stdio.h>", ())
    print(f"Result for '<stdio.h>': {result}")
except Exception as e:
    print(f"Exception: {e}")

print("\nTest 5: Testing the property-based test")
from hypothesis import given, strategies as st

@given(st.lists(st.text(min_size=1)), st.text())
def test_resolve_depend_handles_all_depend_strings(include_dirs, depend):
    result = resolve_depend(depend, tuple(include_dirs))
    assert result is None or isinstance(result, str)

try:
    test_resolve_depend_handles_all_depend_strings()
    print("Property test passed!")
except Exception as e:
    print(f"Property test failed: {e}")