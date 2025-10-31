#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.http.cookie import parse_cookie
from hypothesis import given, strategies as st, example

# More targeted test that should fail
@given(st.lists(
    st.tuples(
        st.text(min_size=1, max_size=10, alphabet=' \t\n\r'),  # Only whitespace keys
        st.text(min_size=1, max_size=10, alphabet='abc0123456789')  # Simple values
    ),
    min_size=2,  # At least 2 cookies to test collision
    max_size=5
))
@example([(' ', 'a'), ('\t', 'b')])  # Explicit example
def test_whitespace_key_collision(cookie_list):
    # Create cookie string
    cookie_string = "; ".join(f"{k}={v}" for k, v in cookie_list)

    # Parse it
    result = parse_cookie(cookie_string)

    # Count unique stripped keys
    stripped_keys = [k.strip() for k, v in cookie_list]
    unique_stripped = set(stripped_keys)

    # If all keys strip to empty string and we have multiple cookies
    if all(k == '' for k in stripped_keys) and len(cookie_list) > 1:
        # We expect data loss - only one value should remain
        assert len(result) == 1, f"Expected 1 result, got {len(result)}"
        assert '' in result, "Expected empty key in result"
        # The value should be from the last cookie (due to dict overwriting)
        last_value = cookie_list[-1][1].strip()
        assert result[''] == last_value, f"Expected last value {last_value!r}, got {result['']!r}"
        print(f"✓ Confirmed data loss: {len(cookie_list)} cookies → 1 result")
        print(f"  Input cookies: {cookie_list}")
        print(f"  Final result: {result}")
        return

    # Otherwise check all values are preserved somehow
    for key, val in cookie_list:
        stripped_key = key.strip()
        stripped_val = val.strip()
        if stripped_key or stripped_val:  # Only check if key or val is non-empty after strip
            assert stripped_key in result, f"Key {key!r} (stripped: {stripped_key!r}) missing from result"

if __name__ == "__main__":
    print("Testing whitespace key collisions...\n")

    try:
        test_whitespace_key_collision()
        print("\nAll tests passed!")
    except AssertionError as e:
        print(f"\nTest failed: {e}")