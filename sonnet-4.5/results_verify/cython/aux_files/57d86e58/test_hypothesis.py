import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from pyximport.pyximport import _have_importers, PyxImportMetaFinder, PyImportMetaFinder


@settings(max_examples=500)
@given(st.booleans(), st.booleans())
def test_have_importers_detects_both_types(add_pyx, add_py):
    original_meta_path = sys.meta_path.copy()

    try:
        sys.meta_path = []

        if add_pyx:
            sys.meta_path.append(PyxImportMetaFinder())

        if add_py:
            sys.meta_path.append(PyImportMetaFinder())

        has_py, has_pyx = _have_importers()

        assert has_pyx == add_pyx, \
            f"Expected has_pyx={add_pyx}, got {has_pyx}. meta_path types: {[type(x).__name__ for x in sys.meta_path]}"

        assert has_py == add_py, \
            f"Expected has_py={add_py}, got {has_py}. meta_path types: {[type(x).__name__ for x in sys.meta_path]}"

    finally:
        sys.meta_path[:] = original_meta_path

if __name__ == "__main__":
    test_have_importers_detects_both_types()
    print("Test completed")