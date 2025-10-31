#!/usr/bin/env python3
"""Test to understand parse_bytes behavior with empty strings"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

import dask.utils

def test_parse_bytes():
    """Test parse_bytes with various inputs including edge cases"""
    test_cases = ['', ' ', '\r', '\n', '\t', 'MB', '100', '100 MB']

    for case in test_cases:
        print(f"Testing parse_bytes({repr(case)})...")
        try:
            result = dask.utils.parse_bytes(case)
            print(f"  Result: {result}")
        except ValueError as e:
            print(f"  ValueError: {e}")
        except Exception as e:
            print(f"  {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_parse_bytes()