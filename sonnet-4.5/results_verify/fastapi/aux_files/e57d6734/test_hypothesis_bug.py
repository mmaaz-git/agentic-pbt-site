#!/usr/bin/env python3
"""Test the reported bug using hypothesis"""

from hypothesis import given, strategies as st
from fastapi.dependencies.utils import get_typed_annotation
import keyword
import traceback

# Test with hypothesis
@given(annotation_str=st.sampled_from(keyword.kwlist))
def test_get_typed_annotation_handles_keywords(annotation_str):
    """Test that get_typed_annotation handles Python keywords as string annotations"""
    print(f"Testing with keyword: {annotation_str}")
    globalns = {}
    try:
        result = get_typed_annotation(annotation_str, globalns)
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Exception: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise

# Manual test with specific keywords
def test_manual():
    print("\n=== Manual test with specific keywords ===")
    test_keywords = ['if', 'class', 'def', 'while', 'for', 'return', 'lambda']

    for kw in test_keywords:
        print(f"\nTesting keyword: '{kw}'")
        globalns = {}
        try:
            result = get_typed_annotation(kw, globalns)
            print(f"  Success: {result}")
        except SyntaxError as e:
            print(f"  SyntaxError: {e}")
        except Exception as e:
            print(f"  Other exception ({type(e).__name__}): {e}")

if __name__ == "__main__":
    # Run manual test first
    test_manual()

    # Then run hypothesis test
    print("\n=== Running hypothesis test ===")
    try:
        test_get_typed_annotation_handles_keywords()
    except:
        print("Hypothesis test framework execution failed")