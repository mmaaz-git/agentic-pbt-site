#!/usr/bin/env python3
"""Test script to reproduce the parse_default bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

print("=== Testing property-based test ===")
from hypothesis import given, strategies as st, settings
from Cython.Tempita import Template, TemplateError

@given(st.text(alphabet=' \t', min_size=0, max_size=5))
@settings(max_examples=100)
def test_parse_default_whitespace_only(whitespace):
    content = f"{{{{default{whitespace}}}}}"

    try:
        template = Template(content)
        assert False, f"Should raise TemplateError for content: {repr(content)}"
    except Exception as e:
        assert isinstance(e, TemplateError), f"Should be TemplateError, got {type(e).__name__} for content: {repr(content)}"
        assert 'no = found' in str(e) or 'Not a valid variable name' in str(e), f"Wrong error message: {e}"

# Run the test
try:
    test_parse_default_whitespace_only()
    print("Property-based test passed (unexpected!)")
except AssertionError as e:
    print(f"Property-based test failed as expected: {e}")
except Exception as e:
    print(f"Unexpected error in property-based test: {type(e).__name__}: {e}")

print("\n=== Testing specific reproduction case ===")
content = "{{default }}"

try:
    template = Template(content)
    print(f"Template created successfully (unexpected!)")
except IndexError as e:
    print(f"IndexError: {e}")
    print(f"Expected: TemplateError with message about missing '='")
    print(f"Actual: IndexError: list index out of range")
except TemplateError as e:
    print(f"Got TemplateError as expected: {e}")
except Exception as e:
    print(f"Got unexpected exception: {type(e).__name__}: {e}")

print("\n=== Testing with empty string after default ===")
content2 = "{{default}}"
try:
    template = Template(content2)
    print(f"Template created successfully for {{default}} (unexpected!)")
except IndexError as e:
    print(f"IndexError for {{default}}: {e}")
except TemplateError as e:
    print(f"TemplateError for {{default}}: {e}")
except Exception as e:
    print(f"Other error for {{default}}: {type(e).__name__}: {e}")