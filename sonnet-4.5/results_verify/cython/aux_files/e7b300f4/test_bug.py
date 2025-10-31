#!/usr/bin/env python3
"""Test the reported bug with Cython.Tempita bytes handling"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

print("Test 1: Simple manual reproduction")
print("-" * 40)
try:
    from Cython.Tempita import Template

    bytes_content = b"Hello {{name}}"
    print(f"Creating template with bytes content: {bytes_content}")
    template = Template(bytes_content)
    print("Template created successfully")

    result = template.substitute({'name': 'World'})
    print(f"Result: {result}")
    print(f"Result type: {type(result)}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n\nTest 2: Property-based test with hypothesis")
print("-" * 40)
try:
    from hypothesis import given, assume, strategies as st

    @given(st.text(min_size=1, max_size=100))
    def test_template_bytes_content_handling(value):
        assume('\x00' not in value)

        content_bytes = value.encode('utf-8')
        template = Template(content_bytes)

        result = template.substitute({})
        assert isinstance(result, bytes) or isinstance(result, str)
        return True

    # Run a few test cases
    test_template_bytes_content_handling()
    print("Hypothesis test passed (should have failed according to report)")
except Exception as e:
    print(f"Hypothesis test failed as expected: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n\nTest 3: More comprehensive tests")
print("-" * 40)
# Test with string content to see if that works
try:
    string_content = "Hello {{name}}"
    print(f"Creating template with string content: {string_content}")
    template = Template(string_content)
    print("Template created successfully")

    result = template.substitute({'name': 'World'})
    print(f"Result: {result}")
    print(f"Result type: {type(result)}")
except Exception as e:
    print(f"Error with string: {type(e).__name__}: {e}")

# Test with empty bytes
print("\nTest with empty bytes:")
try:
    template = Template(b"")
    result = template.substitute({})
    print(f"Empty bytes result: {result}, type: {type(result)}")
except Exception as e:
    print(f"Error with empty bytes: {type(e).__name__}: {e}")

# Test with bytes without template syntax
print("\nTest with plain bytes (no template syntax):")
try:
    template = Template(b"Hello World")
    result = template.substitute({})
    print(f"Plain bytes result: {result}, type: {type(result)}")
except Exception as e:
    print(f"Error with plain bytes: {type(e).__name__}: {e}")