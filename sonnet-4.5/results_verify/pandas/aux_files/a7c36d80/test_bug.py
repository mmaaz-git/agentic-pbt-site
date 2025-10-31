from pandas.core.indexers.utils import length_of_indexer

indexer = range(0, 1, 2)
computed = length_of_indexer(indexer)
actual = len(indexer)

print(f"length_of_indexer(range(0, 1, 2)) = {computed}")
print(f"len(range(0, 1, 2)) = {actual}")
print(f"Are they equal? {computed == actual}")

if computed != actual:
    print(f"ERROR: Mismatch! Expected {actual}, got {computed}")