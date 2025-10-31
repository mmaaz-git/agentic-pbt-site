#!/usr/bin/env python3
"""Test existing validation cases mentioned in the bug report"""

import numpy as np
from xarray.plot.utils import _rescale_imshow_rgb

def test_vmax_none_with_invalid_vmin():
    """Test when vmax is None but vmin is too large"""
    print("Testing: vmax=None, vmin=500 (should fail since default vmax is 1 or 255)")
    darray = np.random.uniform(0, 100, (10, 10, 3)).astype('f8')

    try:
        result = _rescale_imshow_rgb(darray, vmin=500, vmax=None, robust=False)
        print(f"  No error raised! Result shape: {result.shape}")
    except ValueError as e:
        print(f"  ValueError correctly raised: {e}")

def test_vmin_none_with_invalid_vmax():
    """Test when vmin is None but vmax is negative"""
    print("\nTesting: vmin=None, vmax=-10 (should fail since default vmin is 0)")
    darray = np.random.uniform(0, 100, (10, 10, 3)).astype('f8')

    try:
        result = _rescale_imshow_rgb(darray, vmin=None, vmax=-10, robust=False)
        print(f"  No error raised! Result shape: {result.shape}")
    except ValueError as e:
        print(f"  ValueError correctly raised: {e}")

def test_both_provided_invalid():
    """Test when both vmin and vmax are provided but vmin > vmax"""
    print("\nTesting: vmin=100, vmax=50 (both provided, vmin > vmax)")
    darray = np.random.uniform(0, 100, (10, 10, 3)).astype('f8')

    try:
        result = _rescale_imshow_rgb(darray, vmin=100, vmax=50, robust=False)
        print(f"  No error raised! Result shape: {result.shape}")
        print(f"  Result min/max: {np.min(result)}, {np.max(result)}")
    except ValueError as e:
        print(f"  ValueError correctly raised: {e}")

def test_integer_data():
    """Test with integer data (default vmax should be 255)"""
    print("\nTesting with integer data: vmax=None, vmin=300")
    darray = np.random.randint(0, 100, (10, 10, 3), dtype=np.int32)

    try:
        result = _rescale_imshow_rgb(darray, vmin=300, vmax=None, robust=False)
        print(f"  No error raised! Result shape: {result.shape}")
    except ValueError as e:
        print(f"  ValueError correctly raised: {e}")

def test_edge_case_equal():
    """Test when vmin == vmax"""
    print("\nTesting: vmin=50, vmax=50 (equal values)")
    darray = np.random.uniform(0, 100, (10, 10, 3)).astype('f8')

    try:
        result = _rescale_imshow_rgb(darray, vmin=50, vmax=50, robust=False)
        print(f"  No error raised! Result shape: {result.shape}")
        print(f"  Result contains: {np.unique(result)}")
    except (ValueError, ZeroDivisionError) as e:
        print(f"  Error raised: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_vmax_none_with_invalid_vmin()
    test_vmin_none_with_invalid_vmax()
    test_both_provided_invalid()
    test_integer_data()
    test_edge_case_equal()