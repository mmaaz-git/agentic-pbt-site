#!/usr/bin/env python3
"""Test the extended_iglob duplicate behavior"""

import os
import tempfile
from pathlib import Path
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Build.Dependencies import extended_iglob

def test_brace_expansion_duplicates():
    """Test if brace expansion with duplicates returns duplicates"""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir)

        # Create test files
        (test_dir / 'test.py').write_text('test')
        (test_dir / 'test.pyx').write_text('test')

        orig_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            # Test pattern with duplicate 'py' in brace expansion
            pattern = 'test.{py,pyx,py}'
            results = list(extended_iglob(pattern))

            print(f"Pattern: {pattern}")
            print(f"Results: {results}")
            print(f"Number of results: {len(results)}")

            unique_results = list(set(results))
            print(f"Unique results: {unique_results}")
            print(f"Number of unique results: {len(unique_results)}")

            if len(results) != len(unique_results):
                print("\nDUPLICATES FOUND!")
                print(f"The function returned duplicates: {results}")
            else:
                print("\nNo duplicates found.")

        finally:
            os.chdir(orig_cwd)

def test_recursive_glob_duplicates():
    """Test if recursive glob prevents duplicates (for comparison)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir)

        # Create nested structure
        (test_dir / 'test.py').write_text('test')
        (test_dir / 'sub').mkdir()
        (test_dir / 'sub' / 'test.py').write_text('test')

        orig_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            # Test recursive pattern
            pattern = '**/test.py'
            results = list(extended_iglob(pattern))

            print(f"\nRecursive pattern: {pattern}")
            print(f"Results: {results}")
            print(f"Number of results: {len(results)}")

            unique_results = list(set(results))
            print(f"Unique results: {unique_results}")

            if len(results) != len(unique_results):
                print("DUPLICATES in recursive glob!")
            else:
                print("No duplicates in recursive glob (as expected)")

        finally:
            os.chdir(orig_cwd)

def test_combined_patterns():
    """Test pattern that combines both features"""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir)

        # Create test files
        (test_dir / 'test.py').write_text('test')
        (test_dir / 'test.pyx').write_text('test')
        (test_dir / 'sub').mkdir()
        (test_dir / 'sub' / 'test.py').write_text('test')

        orig_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            # Combined pattern
            pattern = '**/test.{py,py}'
            results = list(extended_iglob(pattern))

            print(f"\nCombined pattern: {pattern}")
            print(f"Results: {results}")

            unique_results = list(set(results))
            if len(results) != len(unique_results):
                print(f"DUPLICATES in combined pattern! {len(results)} vs {len(unique_results)} unique")
            else:
                print("No duplicates in combined pattern")

        finally:
            os.chdir(orig_cwd)

if __name__ == "__main__":
    print("Testing extended_iglob duplicate behavior\n")
    print("="*50)
    test_brace_expansion_duplicates()
    test_recursive_glob_duplicates()
    test_combined_patterns()