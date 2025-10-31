#!/usr/bin/env python3
"""Test script to reproduce the eliminate_quotes bug"""

import sys
import traceback

def test_simple_unpaired_quotes():
    """Test the simple reproduction case"""
    from numpy.f2py import symbolic

    print("Testing with single double-quote character...")
    try:
        s = '"'
        new_s, mapping = symbolic.eliminate_quotes(s)
        print(f"Success: {s!r} -> {new_s!r}")
    except AssertionError as e:
        print(f"AssertionError caught for input: {s!r}")
        traceback.print_exc()
    except Exception as e:
        print(f"Other error: {type(e).__name__}: {e}")
        traceback.print_exc()

    print("\nTesting with single single-quote character...")
    try:
        s = "'"
        new_s, mapping = symbolic.eliminate_quotes(s)
        print(f"Success: {s!r} -> {new_s!r}")
    except AssertionError as e:
        print(f"AssertionError caught for input: {s!r}")
        traceback.print_exc()
    except Exception as e:
        print(f"Other error: {type(e).__name__}: {e}")
        traceback.print_exc()

    print("\nTesting with paired quotes (should work)...")
    try:
        s = '"hello"'
        new_s, mapping = symbolic.eliminate_quotes(s)
        print(f"Success: {s!r} -> {new_s!r}, mapping: {mapping}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_unpaired_quotes()