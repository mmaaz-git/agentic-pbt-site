#!/usr/bin/env python3
"""Test script to reproduce the reported bug in numpy.f2py.symbolic.replace_parenthesis"""

import traceback
import sys

def test_basic_functionality():
    """Test that the function works correctly with matched parentheses"""
    import numpy.f2py.symbolic as symbolic

    print("Testing basic functionality with matched parentheses...")

    # Test with matched parentheses
    test_cases = [
        "(a)",
        "[b]",
        "{c}",
        "(a(b)c)",
        "x + (y * z)",
    ]

    for test in test_cases:
        try:
            result, mapping = symbolic.replace_parenthesis(test)
            restored = symbolic.unreplace_parenthesis(result, mapping)
            print(f"  Input: {test!r}")
            print(f"    Result: {result!r}, Mapping: {mapping}")
            print(f"    Restored: {restored!r}")
            print(f"    Round-trip OK: {test == restored}")
        except Exception as e:
            print(f"  Input: {test!r} - FAILED: {e}")
    print()

def test_unmatched_bracket():
    """Test the reported bug with unmatched opening bracket"""
    import numpy.f2py.symbolic as symbolic

    print("Testing with unmatched opening bracket '[' ...")
    try:
        result, mapping = symbolic.replace_parenthesis('[')
        print(f"  Result: {result!r}, Mapping: {mapping}")
    except RecursionError as e:
        print(f"  RecursionError raised as reported!")
        print(f"  Error type: {type(e).__name__}")
        # Print just the last few lines of traceback to avoid clutter
        tb_lines = traceback.format_exc().split('\n')
        print("  Last few traceback lines:")
        for line in tb_lines[-10:-1]:
            if line:
                print(f"    {line}")
    except Exception as e:
        print(f"  Different error raised: {type(e).__name__}: {e}")
    print()

def test_counter_corruption():
    """Test if COUNTER gets corrupted after RecursionError"""
    import numpy.f2py.symbolic as symbolic

    print("Testing COUNTER corruption after RecursionError...")

    # First cause the RecursionError
    print("  1. Triggering RecursionError with '['...")
    try:
        symbolic.replace_parenthesis('[')
    except RecursionError:
        print("     RecursionError triggered successfully")
    except Exception as e:
        print(f"     Different error: {type(e).__name__}: {e}")

    # Now try to use the function normally
    print("  2. Trying to use replace_parenthesis('(a)') after RecursionError...")
    try:
        result, mapping = symbolic.replace_parenthesis('(a)')
        print(f"     Success! Result: {result!r}, Mapping: {mapping}")
    except StopIteration:
        print("     StopIteration raised - COUNTER is corrupted as reported!")
    except Exception as e:
        print(f"     Different error: {type(e).__name__}: {e}")
    print()

def test_hypothesis_property():
    """Run the hypothesis test from the bug report"""
    print("Running the hypothesis property test...")
    try:
        from hypothesis import given, strategies as st
        import numpy.f2py.symbolic as symbolic

        @given(st.text(alphabet='()[]{}', min_size=1, max_size=30))
        def test_replace_unreplace_parenthesis_roundtrip(s):
            s_no_parens, d = symbolic.replace_parenthesis(s)
            s_restored = symbolic.unreplace_parenthesis(s_no_parens, d)
            assert s == s_restored, f"Round-trip failed for {s!r}: got {s_restored!r}"

        # Try to run the test with a small sample
        print("  Testing with specific failing input '[' from bug report...")
        try:
            test_replace_unreplace_parenthesis_roundtrip('[')
            print("    Test passed (unexpected!)")
        except RecursionError:
            print("    RecursionError as expected")
        except AssertionError as e:
            print(f"    Assertion failed: {e}")
        except Exception as e:
            print(f"    Other error: {type(e).__name__}: {e}")

    except ImportError:
        print("  Hypothesis not installed, skipping property test")
    print()

def test_other_unmatched():
    """Test other unmatched parentheses cases"""
    import numpy.f2py.symbolic as symbolic

    print("Testing other unmatched parentheses...")
    test_cases = [
        '(',
        ')',
        ']',
        '{',
        '}',
        '(]',
        '[)',
        '(((',
        ')))',
    ]

    for test in test_cases:
        print(f"  Testing {test!r}...")
        try:
            result, mapping = symbolic.replace_parenthesis(test)
            print(f"    Result: {result!r}, Mapping: {mapping}")
        except RecursionError:
            print(f"    RecursionError!")
        except ValueError as e:
            print(f"    ValueError: {e}")
        except Exception as e:
            print(f"    {type(e).__name__}: {e}")
    print()

if __name__ == "__main__":
    print("=" * 60)
    print("Bug Report Verification for numpy.f2py.symbolic.replace_parenthesis")
    print("=" * 60)
    print()

    test_basic_functionality()
    test_unmatched_bracket()
    test_counter_corruption()
    test_hypothesis_property()
    test_other_unmatched()

    print("=" * 60)
    print("Testing complete")
    print("=" * 60)