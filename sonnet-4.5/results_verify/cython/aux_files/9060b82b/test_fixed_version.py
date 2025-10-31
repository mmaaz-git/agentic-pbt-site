import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from pyximport.pyximport import PyImportMetaFinder, PyxImportMetaFinder

def _have_importers_fixed():
    has_py_importer = False
    has_pyx_importer = False
    for importer in sys.meta_path:
        if isinstance(importer, PyImportMetaFinder):
            has_py_importer = True
        elif isinstance(importer, PyxImportMetaFinder):
            has_pyx_importer = True

    return has_py_importer, has_pyx_importer

# Test the fixed version
original_meta_path = sys.meta_path.copy()
try:
    # Test case 1: PyImportMetaFinder only
    sys.meta_path = []
    sys.meta_path.append(PyImportMetaFinder())
    has_py, has_pyx = _have_importers_fixed()
    print(f"Test 1 (PyImport only): has_py={has_py}, has_pyx={has_pyx}")
    assert has_py == True and has_pyx == False, f"Test 1 failed"

    # Test case 2: PyxImportMetaFinder only
    sys.meta_path = []
    sys.meta_path.append(PyxImportMetaFinder())
    has_py, has_pyx = _have_importers_fixed()
    print(f"Test 2 (PyxImport only): has_py={has_py}, has_pyx={has_pyx}")
    assert has_py == False and has_pyx == True, f"Test 2 failed"

    # Test case 3: Both
    sys.meta_path = []
    sys.meta_path.append(PyImportMetaFinder())
    sys.meta_path.append(PyxImportMetaFinder())
    has_py, has_pyx = _have_importers_fixed()
    print(f"Test 3 (Both): has_py={has_py}, has_pyx={has_pyx}")
    assert has_py == True and has_pyx == True, f"Test 3 failed"

    # Test case 4: Neither
    sys.meta_path = []
    has_py, has_pyx = _have_importers_fixed()
    print(f"Test 4 (Neither): has_py={has_py}, has_pyx={has_pyx}")
    assert has_py == False and has_pyx == False, f"Test 4 failed"

    print("\nAll tests passed! The fix works correctly.")

finally:
    sys.meta_path[:] = original_meta_path