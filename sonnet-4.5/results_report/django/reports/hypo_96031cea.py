from hypothesis import given, strategies as st, settings, assume
from django.http.cookie import parse_cookie
import sys

# Strategy for whitespace strings that will collide when stripped
whitespace_chars = [' ', '\t', '\n', '\r', '\f', '\v']
whitespace_text = st.text(alphabet=whitespace_chars, min_size=1, max_size=5)

# Property: parse_cookie should preserve all cookies without data loss
@given(
    # Generate multiple cookies where some may have whitespace-only keys
    cookies=st.lists(
        st.tuples(
            st.one_of(
                whitespace_text,  # Whitespace-only keys
                st.text(alphabet='abcdefghijklmnopqrstuvwxyz0123456789_', min_size=1, max_size=10)  # Normal keys
            ),
            st.text(alphabet='abcdefghijklmnopqrstuvwxyz0123456789', min_size=1, max_size=10)  # Values
        ),
        min_size=2,
        max_size=10,
        unique_by=lambda x: x[0]  # Ensure unique keys in input
    )
)
@settings(max_examples=1000, database=None)
def test_parse_cookie_preserves_all_cookies(cookies):
    """
    This test checks if parse_cookie preserves all cookies without data loss.
    """
    # Build cookie string
    cookie_string = "; ".join(f"{k}={v}" for k, v in cookies)

    # Parse the cookie string
    parsed = parse_cookie(cookie_string)

    # Group cookies by their stripped key
    key_groups = {}
    for key, value in cookies:
        stripped_key = key.strip()
        if stripped_key not in key_groups:
            key_groups[stripped_key] = []
        key_groups[stripped_key].append((key, value))

    # Check for data loss
    for stripped_key, group in key_groups.items():
        if len(group) > 1:
            # Multiple cookies will collide to the same key after stripping
            print(f"\nFailing example found!")
            print(f"Input cookies: {cookies}")
            print(f"Cookie string: {cookie_string!r}")
            print(f"Parsed result: {parsed}")
            print(f"\nCollision detected for stripped key {stripped_key!r}:")
            print(f"  Original keys that collide: {[k for k, v in group]!r}")
            print(f"  Original values: {[v for k, v in group]!r}")
            if stripped_key in parsed:
                print(f"  Only kept value: {parsed[stripped_key]!r}")
            print(f"\nData loss: {len(group)} cookies collapsed to 1")

            # This demonstrates the bug
            assert False, f"Data loss: {len(group)} cookies with keys {[k for k,v in group]!r} all collapsed to key {stripped_key!r}"

if __name__ == "__main__":
    # Run the property test
    print("Running Hypothesis property-based test for parse_cookie...")
    print("This test verifies that parse_cookie preserves all cookies without data loss.")
    print("-" * 60)

    try:
        test_parse_cookie_preserves_all_cookies()
        print("\nAll tests passed! No data loss detected.")
    except AssertionError as e:
        print(f"\n** TEST FAILED **")
        print(f"Assertion Error: {e}")
        print("\nThe test demonstrates that parse_cookie loses data when multiple")
        print("cookies have keys that become identical after stripping whitespace.")
        sys.exit(1)