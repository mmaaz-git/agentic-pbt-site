import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from pyximport.pyximport import _have_importers, PyImportMetaFinder, PyxImportMetaFinder
from hypothesis import given, strategies as st, settings


@settings(max_examples=50)
@given(
    add_py=st.booleans(),
    add_pyx=st.booleans()
)
def test_have_importers_detects_all_finders(add_py, add_pyx):
    original_meta_path = sys.meta_path.copy()
    try:
        sys.meta_path = []

        if add_py:
            sys.meta_path.append(PyImportMetaFinder())
        if add_pyx:
            sys.meta_path.append(PyxImportMetaFinder())

        has_py, has_pyx = _have_importers()

        assert has_py == add_py, f"Expected has_py={add_py}, got {has_py}"
        assert has_pyx == add_pyx, f"Expected has_pyx={add_pyx}, got {has_pyx}"

    finally:
        sys.meta_path[:] = original_meta_path


if __name__ == "__main__":
    test_have_importers_detects_all_finders()
    print("Test completed successfully!")