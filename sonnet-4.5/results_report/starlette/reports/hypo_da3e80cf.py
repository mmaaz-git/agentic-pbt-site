#!/usr/bin/env python3
"""
Property-based test that fails for the unsigned_long_long bug.
"""

from hypothesis import given, strategies as st
from numpy.f2py import capi_maps

@given(st.sampled_from(list(capi_maps.c2capi_map.keys())))
def test_c2capi_keys_have_c2py_mapping(ctype):
    assert ctype in capi_maps.c2py_map, \
        f"C type {ctype!r} in c2capi_map but not in c2py_map"

if __name__ == "__main__":
    # Run the test
    test_c2capi_keys_have_c2py_mapping()