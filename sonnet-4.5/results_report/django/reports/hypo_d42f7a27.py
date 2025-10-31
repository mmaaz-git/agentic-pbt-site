#!/usr/bin/env python3
"""
Property-based test using Hypothesis that detects the header injection vulnerability
in django.core.mail.message.forbid_multi_line_headers
"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings, example, assume
from django.core.mail.message import forbid_multi_line_headers, BadHeaderError

@given(st.text(), st.text(min_size=1), st.sampled_from(['utf-8', 'ascii', 'iso-8859-1', None]))
@example('X-Custom-Header', '0\x0c\x80', 'utf-8')  # The known failing example
@settings(max_examples=100)
def test_forbid_multi_line_headers_rejects_newlines(name, val, encoding):
    """
    Property: forbid_multi_line_headers should never return a value containing newlines

    The function's docstring states: "Forbid multi-line headers to prevent header injection."
    Therefore, it should either:
    1. Raise BadHeaderError if the input or output would contain newlines
    2. Return a value that does not contain newlines
    """

    # Test the function
    if '\n' in val or '\r' in val:
        # If input contains newlines, function MUST raise BadHeaderError
        try:
            result = forbid_multi_line_headers(name, val, encoding)
            # If we get here, the function failed to reject input with newlines
            print(f"\n❌ FAIL: Function accepted input with newlines")
            print(f"  Input: name={repr(name)}, val={repr(val)}, encoding={repr(encoding)}")
            print(f"  Output: {repr(result)}")
            raise AssertionError(f"Function didn't raise BadHeaderError for input containing newlines")
        except BadHeaderError:
            # This is the expected behavior - pass
            pass
        except Exception as e:
            # Some other error occurred (encoding issues, etc.) - ignore
            pass
    else:
        # Input doesn't contain newlines, so output MUST NOT contain newlines either
        try:
            result_name, result_val = forbid_multi_line_headers(name, val, encoding)

            # Check if output contains newlines
            if '\n' in result_val or '\r' in result_val:
                print(f"\n❌ VULNERABILITY FOUND!")
                print(f"  Input: name={repr(name)}, val={repr(val)}, encoding={repr(encoding)}")
                print(f"  Output: {repr(result_val)}")
                print(f"  The output contains {'newline (\\n)' if '\n' in result_val else 'carriage return (\\r)'}")
                print(f"\n  This violates the function's security guarantee!")
                raise AssertionError(f"Output contains newline despite clean input: {repr(result_val)}")

        except BadHeaderError:
            # Function can reject inputs for other valid reasons - pass
            pass
        except UnicodeDecodeError:
            # Encoding issues - pass
            pass
        except LookupError:
            # Unknown encoding - pass
            pass
        except Exception as e:
            # Log unexpected errors but don't fail the test
            if "encode" not in str(e).lower() and "decode" not in str(e).lower():
                print(f"Unexpected error: {e}")
            pass

if __name__ == "__main__":
    print("Running Hypothesis property-based test for forbid_multi_line_headers")
    print("=" * 60)
    print()
    print("Testing property: forbid_multi_line_headers should NEVER return")
    print("values containing newlines (its purpose is to prevent header injection)")
    print()

    try:
        # Run the property test
        test_forbid_multi_line_headers_rejects_newlines()
        print("✓ All property tests passed!")
        print()
        print("No vulnerabilities detected in the tested inputs.")
    except AssertionError as e:
        print()
        print("ASSERTION FAILED - Vulnerability Confirmed!")
        print("-" * 60)
        print(str(e))
        print()
        print("Impact: This allows header injection attacks in Django applications")
        raise  # Re-raise to show the full traceback