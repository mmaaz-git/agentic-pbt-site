#!/usr/bin/env python3
"""Test to reproduce the bug from the report"""

from hypothesis import given, strategies as st, settings
from starlette.datastructures import URL

# First, test the simple reproduction case
print("Testing simple reproduction case...")
try:
    url = URL("http://@/path")
    print(f"Created URL: {url}")
    result = url.replace(port=8000)
    print(f"Success! Result: {result}")
except IndexError as e:
    print(f"IndexError caught: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"Other exception: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50 + "\n")

# Now test the hypothesis property test
print("Testing hypothesis property test...")
@given(port=st.integers(min_value=1, max_value=65535))
@settings(max_examples=10)  # Reduced for quick test
def test_url_replace_port_with_empty_hostname(port):
    url = URL("http://@/path")
    result = url.replace(port=port)
    assert isinstance(result, URL)

try:
    test_url_replace_port_with_empty_hostname()
    print("Hypothesis test passed!")
except Exception as e:
    print(f"Hypothesis test failed: {e}")
    import traceback
    traceback.print_exc()