import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from pyximport.pyximport import PyxImportMetaFinder, PyImportMetaFinder

# Test the proposed fix logic
def _have_importers_fixed():
    has_py_importer = False
    has_pyx_importer = False
    for importer in sys.meta_path:
        if isinstance(importer, PyImportMetaFinder):
            has_py_importer = True
        elif isinstance(importer, PyxImportMetaFinder):
            has_pyx_importer = True

    return has_py_importer, has_pyx_importer

# Test all combinations
original_meta_path = sys.meta_path.copy()

print("Testing all four combinations:")
print("-" * 40)

# Test 1: Neither
sys.meta_path = []
has_py, has_pyx = _have_importers_fixed()
print(f"No importers: has_py={has_py}, has_pyx={has_pyx} (expected: False, False)")

# Test 2: Only PyxImportMetaFinder
sys.meta_path = [PyxImportMetaFinder()]
has_py, has_pyx = _have_importers_fixed()
print(f"Only Pyx: has_py={has_py}, has_pyx={has_pyx} (expected: False, True)")

# Test 3: Only PyImportMetaFinder (the failing case)
sys.meta_path = [PyImportMetaFinder()]
has_py, has_pyx = _have_importers_fixed()
print(f"Only Py: has_py={has_py}, has_pyx={has_pyx} (expected: True, False)")

# Test 4: Both
sys.meta_path = [PyxImportMetaFinder(), PyImportMetaFinder()]
has_py, has_pyx = _have_importers_fixed()
print(f"Both: has_py={has_py}, has_pyx={has_pyx} (expected: True, True)")

sys.meta_path[:] = original_meta_path