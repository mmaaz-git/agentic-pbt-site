import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from pyximport.pyximport import _have_importers, PyImportMetaFinder

original_meta_path = sys.meta_path.copy()
try:
    sys.meta_path = []
    sys.meta_path.append(PyImportMetaFinder())

    has_py, has_pyx = _have_importers()
    print(f"has_py={has_py}, has_pyx={has_pyx}")
    assert has_py == True, f"Bug: has_py is {has_py}, expected True"

finally:
    sys.meta_path[:] = original_meta_path

print("If you see this, the bug was not reproduced")