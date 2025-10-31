#!/usr/bin/env python3
"""Run the hypothesis test from the bug report."""

from hypothesis import given, strategies as st, settings
import traceback

@given(st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=10), st.integers())
@settings(max_examples=100)
def test_autocomplete_index_logic(cwords, cword):
    try:
        curr = cwords[cword - 1]
    except IndexError:
        curr = ""

    if cword == 0 and len(cwords) > 0:
        assert curr == "", f"When cword=0, expected curr='', but got {curr!r}"

# Run the test
print("Running hypothesis test from bug report...")
print("=" * 50)

try:
    test_autocomplete_index_logic()
    print("✓ All tests passed!")
except AssertionError as e:
    print(f"✗ Test failed with assertion error:")
    print(f"  {e}")
    traceback.print_exc()
except Exception as e:
    print(f"✗ Test failed with error:")
    print(f"  {e}")
    traceback.print_exc()

# Now test the specific failing case mentioned
print("\n\nTesting specific case: cwords=['0'], cword=0")
print("-" * 40)

cwords = ['0']
cword = 0

try:
    curr = cwords[cword - 1]
except IndexError:
    curr = ""

print(f"cwords = {cwords}")
print(f"cword = {cword}")
print(f"curr = {curr!r}")

if cword == 0 and len(cwords) > 0:
    if curr == "":
        print("✓ Test passed: curr is empty as expected")
    else:
        print(f"✗ Test failed: When cword=0, expected curr='', but got {curr!r}")