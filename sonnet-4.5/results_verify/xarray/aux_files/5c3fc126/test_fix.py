#!/usr/bin/env python3
"""Test if the proposed fix (using tuple instead of list) works"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from functools import lru_cache
from importlib.resources import files

STATIC_FILES = (
    ("xarray.static.html", "icons-svg-inline.html"),
    ("xarray.static.css", "style.css"),
)

# Original implementation (returns list)
@lru_cache(None)
def _load_static_files_original():
    """Original implementation - returns mutable list"""
    return [
        files(package).joinpath(resource).read_text(encoding="utf-8")
        for package, resource in STATIC_FILES
    ]

# Fixed implementation (returns tuple)
@lru_cache(None)
def _load_static_files_fixed():
    """Fixed implementation - returns immutable tuple"""
    return tuple(
        files(package).joinpath(resource).read_text(encoding="utf-8")
        for package, resource in STATIC_FILES
    )

print("Testing the proposed fix...")
print("=" * 60)

# Test the fixed version
print("Testing FIXED version (returns tuple):")
first_call = _load_static_files_fixed()
print(f"Type: {type(first_call)}")
print(f"Number of elements: {len(first_call)}")

second_call = _load_static_files_fixed()
print(f"Same object? {first_call is second_call}")

try:
    # Try to mutate it
    second_call[0] = "MUTATED"
    print("ERROR: Should not be able to mutate a tuple!")
except TypeError as e:
    print(f"Good! Cannot mutate tuple: {e}")

# Verify the cache is still intact
third_call = _load_static_files_fixed()
print(f"Third call still has original content: {third_call[0][:50]}")

print("\n" + "=" * 60)
print("Testing how the fixed version would be used in the actual code...")

# Check how it's used in the actual codebase (line 305 of formatting_html.py)
# icons_svg, css_style = _load_static_files()

# Test unpacking still works with tuple
icons_svg, css_style = _load_static_files_fixed()
print(f"Unpacking works: icons_svg type={str(type(icons_svg))[:20]}..., css_style type={str(type(css_style))[:20]}...")

# Test indexing still works
result = _load_static_files_fixed()
print(f"Indexing works: result[0][:50] = {result[0][:50]}")
print(f"Indexing works: result[1][:50] = {result[1][:50]}")

print("\n" + "=" * 60)
print("Conclusion: The fix (using tuple instead of list) would:")
print("1. Prevent cache mutation (tuples are immutable)")
print("2. Maintain compatibility with existing code (unpacking and indexing work)")
print("3. Keep the same performance benefits of caching")