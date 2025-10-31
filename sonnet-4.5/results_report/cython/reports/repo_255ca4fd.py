import sys
import pyximport

original_meta_path = sys.meta_path.copy()

print("Initial count:", len([i for i in sys.meta_path if isinstance(i, pyximport.PyImportMetaFinder)]))

pyximport.install(pyimport=True, pyximport=False)
print("After 1st install:", len([i for i in sys.meta_path if isinstance(i, pyximport.PyImportMetaFinder)]))

pyximport.install(pyimport=True, pyximport=False)
print("After 2nd install:", len([i for i in sys.meta_path if isinstance(i, pyximport.PyImportMetaFinder)]))

pyximport.install(pyimport=True, pyximport=False)
print("After 3rd install:", len([i for i in sys.meta_path if isinstance(i, pyximport.PyImportMetaFinder)]))

sys.meta_path = original_meta_path