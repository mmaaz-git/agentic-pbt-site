#!/usr/bin/env python3
"""Test script to reproduce the Kleene operations bug"""

import sys
import traceback
from pandas.core.ops import kleene_and, kleene_or, kleene_xor

def test_kleene_and_with_both_none():
    """Test kleene_and with both masks as None"""
    print("Testing kleene_and(False, False, None, None)...")
    try:
        result, mask = kleene_and(False, False, None, None)
        print(f"Result: {result}, Mask: {mask}")
    except RecursionError as e:
        print(f"RecursionError encountered: {str(e)[:100]}...")
        return "RecursionError"
    except Exception as e:
        print(f"Other exception: {type(e).__name__}: {e}")
        return str(type(e).__name__)
    return "Success"

def test_kleene_or_with_both_none():
    """Test kleene_or with both masks as None"""
    print("\nTesting kleene_or(True, False, None, None)...")
    try:
        result, mask = kleene_or(True, False, None, None)
        print(f"Result: {result}, Mask: {mask}")
    except RecursionError as e:
        print(f"RecursionError encountered: {str(e)[:100]}...")
        return "RecursionError"
    except Exception as e:
        print(f"Other exception: {type(e).__name__}: {e}")
        return str(type(e).__name__)
    return "Success"

def test_kleene_xor_with_both_none():
    """Test kleene_xor with both masks as None"""
    print("\nTesting kleene_xor(True, False, None, None)...")
    try:
        result, mask = kleene_xor(True, False, None, None)
        print(f"Result: {result}, Mask: {mask}")
    except RecursionError as e:
        print(f"RecursionError encountered: {str(e)[:100]}...")
        return "RecursionError"
    except Exception as e:
        print(f"Other exception: {type(e).__name__}: {e}")
        return str(type(e).__name__)
    return "Success"

def test_hypothesis_case():
    """Test the specific hypothesis case from the bug report"""
    print("\nTesting with hypothesis-style inputs...")
    from hypothesis import given, strategies as st, settings
    from pandas._libs import missing as libmissing

    bool_or_na = st.one_of(st.booleans(), st.just(libmissing.NA))

    @given(left=bool_or_na, right=bool_or_na)
    @settings(max_examples=5)
    def test_kleene_and_commutativity_scalars(left, right):
        try:
            result1, mask1 = kleene_and(left, right, None, None)
            result2, mask2 = kleene_and(right, left, None, None)
            assert result1 == result2
            assert mask1 == mask2
            return "Success"
        except RecursionError:
            return "RecursionError"
        except Exception as e:
            return f"Exception: {type(e).__name__}"

    try:
        test_kleene_and_commutativity_scalars()
        print("Hypothesis test passed")
    except Exception as e:
        print(f"Hypothesis test failed: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Kleene operations with both masks as None")
    print("=" * 60)

    results = []

    # Set recursion limit to something reasonable to avoid hanging
    sys.setrecursionlimit(100)

    results.append(("kleene_and", test_kleene_and_with_both_none()))
    results.append(("kleene_or", test_kleene_or_with_both_none()))
    results.append(("kleene_xor", test_kleene_xor_with_both_none()))

    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    for func, result in results:
        print(f"{func}: {result}")

    # Test hypothesis
    print("\n" + "=" * 60)
    print("Testing with Hypothesis framework:")
    print("=" * 60)
    test_hypothesis_case()