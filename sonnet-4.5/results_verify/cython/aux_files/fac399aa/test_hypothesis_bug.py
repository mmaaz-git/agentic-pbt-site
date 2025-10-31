#!/usr/bin/env python3
"""Hypothesis test to reproduce the Cython.Distutils.Extension bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from Cython.Distutils import Extension

valid_identifiers = st.text(min_size=1, max_size=20).filter(lambda x: x.isidentifier())

@given(
    name=valid_identifiers,
    cython_value=st.booleans(),
    pyrex_param=st.sampled_from(['pyrex_include_dirs', 'pyrex_directives', 'pyrex_gdb']),
)
@settings(max_examples=10)
def test_cython_params_preserved_with_pyrex(name, cython_value, pyrex_param):
    kwargs = {
        pyrex_param: [] if 'dirs' in pyrex_param or 'directives' in pyrex_param else False,
        'cython_gdb': cython_value,
    }

    ext = Extension(name, [f"{name}.pyx"], **kwargs)

    assert ext.cython_gdb == cython_value, \
        f"cython_gdb should be {cython_value}, but got {ext.cython_gdb} when {pyrex_param} is also present"

# Run the test
print("Running property-based test...")
try:
    test_cython_params_preserved_with_pyrex()
    print("All tests passed!")
except AssertionError as e:
    print(f"Test failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")

# Run the specific failing example
print("\nTesting specific failing example from bug report:")
name = 'A'
cython_value = True
pyrex_param = 'pyrex_include_dirs'

kwargs = {
    pyrex_param: [],
    'cython_gdb': cython_value,
}

try:
    ext = Extension(name, [f"{name}.pyx"], **kwargs)
    print(f"  name: {name}")
    print(f"  cython_value: {cython_value}")
    print(f"  pyrex_param: {pyrex_param}")
    print(f"  Expected cython_gdb: {cython_value}")
    print(f"  Actual cython_gdb: {ext.cython_gdb}")
    assert ext.cython_gdb == cython_value
    print("  Result: PASSED")
except AssertionError:
    print("  Result: FAILED - Bug confirmed!")