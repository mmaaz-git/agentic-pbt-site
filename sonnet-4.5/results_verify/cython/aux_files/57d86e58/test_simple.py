import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from pyximport.pyximport import _have_importers, PyImportMetaFinder

original_meta_path = sys.meta_path.copy()

sys.meta_path = [PyImportMetaFinder()]
has_py, has_pyx = _have_importers()

print(f"meta_path contains: PyImportMetaFinder")
print(f"_have_importers() returned: has_py={has_py}, has_pyx={has_pyx}")
print(f"Expected: has_py=True, has_pyx=False")
print(f"Bug: has_py={has_py} (should be True)")

sys.meta_path[:] = original_meta_path