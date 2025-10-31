#!/usr/bin/env python3
"""Test to reproduce the multiple else clauses bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template, TemplateError

# Test 1: Multiple else clauses
print("Test 1: Multiple else clauses")
content1 = """
{{if x}}
true_branch
{{else}}
first_else
{{else}}
second_else
{{endif}}
"""

try:
    template1 = Template(content1)
    result1 = template1.substitute({'x': False})
    print(f"Result with x=False: {repr(result1)}")
    result2 = template1.substitute({'x': True})
    print(f"Result with x=True: {repr(result2)}")
except TemplateError as e:
    print(f"TemplateError raised: {e}")

# Test 2: elif after else
print("\nTest 2: elif after else")
content2 = """
{{if x}}
if_branch
{{else}}
else_branch
{{elif y}}
elif_branch
{{endif}}
"""

try:
    template2 = Template(content2)
    result1 = template2.substitute({'x': False, 'y': True})
    print(f"Result with x=False, y=True: {repr(result1)}")
    result2 = template2.substitute({'x': True, 'y': True})
    print(f"Result with x=True, y=True: {repr(result2)}")
except TemplateError as e:
    print(f"TemplateError raised: {e}")

# Test 3: Property-based test from bug report
print("\nTest 3: Property-based test")
try:
    from hypothesis import given, strategies as st
    import pytest

    @given(st.booleans())
    def test_multiple_else_clauses_rejected(condition):
        content = """
{{if x}}
a
{{else}}
b
{{else}}
c
{{endif}}
"""
        template = Template(content)
        with pytest.raises(TemplateError):
            template.substitute({'x': condition})

    # Run the test
    test_multiple_else_clauses_rejected(True)
    test_multiple_else_clauses_rejected(False)
    print("Property-based test would fail - no TemplateError raised")

except ImportError:
    print("hypothesis not installed, skipping property-based test")
except Exception as e:
    print(f"Property-based test error: {e}")

# Test 4: Standard if/elif/else (should work)
print("\nTest 4: Standard if/elif/else (should work)")
content4 = """
{{if x}}
if_branch
{{elif y}}
elif_branch
{{else}}
else_branch
{{endif}}
"""

try:
    template4 = Template(content4)
    result1 = template4.substitute({'x': True, 'y': False})
    print(f"Result with x=True, y=False: {repr(result1)}")
    result2 = template4.substitute({'x': False, 'y': True})
    print(f"Result with x=False, y=True: {repr(result2)}")
    result3 = template4.substitute({'x': False, 'y': False})
    print(f"Result with x=False, y=False: {repr(result3)}")
except TemplateError as e:
    print(f"TemplateError raised: {e}")