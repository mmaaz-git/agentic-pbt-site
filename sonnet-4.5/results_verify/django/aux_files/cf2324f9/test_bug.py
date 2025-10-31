#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.http.cookie import parse_cookie
from hypothesis import given, strategies as st

# Test 1: Property-based test from bug report
@given(st.dictionaries(
    st.text(min_size=1, max_size=50, alphabet=st.characters(blacklist_characters=set('=;'))),
    st.text(min_size=1, max_size=50, alphabet=st.characters(blacklist_characters=set(';'))),
    min_size=1, max_size=20
))
def test_parse_cookie_preserves_all_cookies(cookies_dict):
    cookie_string = "; ".join(f"{k}={v}" for k, v in cookies_dict.items())
    parsed = parse_cookie(cookie_string)

    for key in cookies_dict.keys():
        assert key.strip() in parsed, f"Key {key!r} (stripped: {key.strip()!r}) not found in parsed cookies"

# Test 2: Specific reproduction case
def test_whitespace_collision():
    print("=== Testing whitespace collision ===")
    cookie_string = " =first; \t=second; \n=third"
    result = parse_cookie(cookie_string)

    print(f"Input:  {cookie_string!r}")
    print(f"Output: {result}")
    print(f"Expected: 3 separate cookies")
    print(f"Actual:   {len(result)} cookie (data loss)")

    # Test the specific case mentioned
    print("\n=== Testing specific failing inputs ===")

    # Test case 1: Single whitespace cookie
    test1 = "\r=0"
    result1 = parse_cookie(test1)
    print(f"Input: {test1!r}")
    print(f"Result: {result1}")

    # Test case 2: Multiple whitespace cookies
    test2 = "\r=0; \n=1"
    result2 = parse_cookie(test2)
    print(f"Input: {test2!r}")
    print(f"Result: {result2}")
    print(f"Data loss: Cookie with key '\\r' and value '0' is lost")

if __name__ == "__main__":
    print("Running reproduction tests...\n")

    # Run the specific reproduction
    test_whitespace_collision()

    # Try to run the property test
    print("\n=== Running property-based test ===")
    try:
        test_parse_cookie_preserves_all_cookies()
        print("Property test passed (no issues found)")
    except AssertionError as e:
        print(f"Property test failed: {e}")
    except Exception as e:
        print(f"Test failed with error: {e}")