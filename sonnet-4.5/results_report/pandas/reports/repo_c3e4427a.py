from pandas.core.indexers.utils import length_of_indexer

target = []
indexer = slice(0, -20, None)

result = length_of_indexer(indexer, target)
actual_length = len(target[indexer])

print(f"length_of_indexer returned: {result}")
print(f"Actual length: {actual_length}")
assert result == actual_length, f"Expected {actual_length}, but got {result}"