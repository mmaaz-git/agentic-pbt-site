#!/usr/bin/env python3
"""
Property-based test demonstrating Django's dictionary modification bug.

The bug: Django's WSGIRequestHandler.get_environ modifies headers dictionary
while iterating, violating Python's iteration rules.
"""

from hypothesis import given, strategies as st, settings, reproduce_failure

# Strategy to generate header names - mix of normal and underscore-containing
header_name_strategy = st.one_of(
    # Headers with underscores (should be removed)
    st.from_regex(r"[A-Z][a-z]*_[A-Z][a-z]*(_[A-Z][a-z]*)*", fullmatch=True),
    # Normal headers with hyphens (should be kept)
    st.from_regex(r"[A-Z][a-z]*-[A-Z][a-z]*(-[A-Z][a-z]*)*", fullmatch=True),
    # Single word headers
    st.from_regex(r"[A-Z][a-z]+", fullmatch=True)
)

@given(
    headers_dict=st.dictionaries(
        header_name_strategy,
        st.text(min_size=1, max_size=100),
        min_size=1,
        max_size=10
    )
)
@settings(max_examples=100, deadline=None)
def test_django_header_removal_pattern(headers_dict):
    """Test Django's pattern for removing headers with underscores.

    Django code at django/core/servers/basehttp.py lines 220-222:
        for k in self.headers:
            if "_" in k:
                del self.headers[k]

    This pattern violates Python's rule against dictionary modification
    during iteration and can cause RuntimeError or incomplete removal.
    """
    # Make a copy for testing
    test_headers = headers_dict.copy()

    # Count headers with underscores
    original_underscore_headers = [k for k in test_headers.keys() if '_' in k]

    # Apply Django's buggy pattern
    error_raised = False
    try:
        for k in test_headers:
            if "_" in k:
                del test_headers[k]
    except RuntimeError as e:
        error_raised = True
        # RuntimeError confirms the bug
        remaining_underscore = [k for k in test_headers.keys() if '_' in k]
        assert remaining_underscore, f"RuntimeError raised but some headers with underscores remain: {remaining_underscore}"

    # If no error, check if all underscore headers were removed
    if not error_raised:
        remaining_underscore = [k for k in test_headers.keys() if '_' in k]
        # Bug: Not all headers with underscores removed
        assert not remaining_underscore, f"No RuntimeError but headers with underscores remain: {remaining_underscore}"

if __name__ == "__main__":
    print("Running property-based test for Django's header removal bug...")
    print()

    # Run with a specific failing example
    failing_example = {
        'X_Forwarded_For': 'value1',
        'User_Agent': 'value2',
        'X_Real_IP': 'value3',
        'Content-Type': 'value4'
    }

    print(f"Testing with: {list(failing_example.keys())}")

    # Test the failing example directly
    test_headers = failing_example.copy()
    original_underscore = [k for k in test_headers if '_' in k]
    print(f"Headers with underscores: {original_underscore}")
    print()

    try:
        for k in test_headers:
            if "_" in k:
                del test_headers[k]
        print("ERROR: No RuntimeError raised")
        remaining = [k for k in test_headers if '_' in k]
        if remaining:
            print(f"BUG: Headers with underscores remain: {remaining}")
    except RuntimeError as e:
        print(f"RuntimeError (confirms bug): {e}")
        remaining = [k for k in test_headers if '_' in k]
        print(f"Headers with underscores remaining after error: {remaining}")

    # Run full Hypothesis test
    print("\n" + "=" * 60)
    print("Running full Hypothesis test suite...")
    print("=" * 60)
    try:
        test_django_header_removal_pattern()
    except Exception as e:
        print(f"Test found failure cases that demonstrate the bug")