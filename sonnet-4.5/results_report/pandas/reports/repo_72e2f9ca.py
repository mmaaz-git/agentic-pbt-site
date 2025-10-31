import pandas.core.indexers as indexers

# Test case 1: range(1, 0, 1)
r = range(1, 0, 1)
print(f"Test 1: range(1, 0, 1)")
print(f"  Python len(range(1, 0, 1)) = {len(r)}")
print(f"  pandas length_of_indexer(range(1, 0, 1)) = {indexers.length_of_indexer(r)}")
print()

# Test case 2: range(10, 0, 2)
r = range(10, 0, 2)
print(f"Test 2: range(10, 0, 2)")
print(f"  Python len(range(10, 0, 2)) = {len(r)}")
print(f"  pandas length_of_indexer(range(10, 0, 2)) = {indexers.length_of_indexer(r)}")
print()

# Test case 3: range(5, 5, 1) - equal start and stop
r = range(5, 5, 1)
print(f"Test 3: range(5, 5, 1) [equal start/stop]")
print(f"  Python len(range(5, 5, 1)) = {len(r)}")
print(f"  pandas length_of_indexer(range(5, 5, 1)) = {indexers.length_of_indexer(r)}")
print()

# Test case 4: range(100, 50, 3) - larger range
r = range(100, 50, 3)
print(f"Test 4: range(100, 50, 3)")
print(f"  Python len(range(100, 50, 3)) = {len(r)}")
print(f"  pandas length_of_indexer(range(100, 50, 3)) = {indexers.length_of_indexer(r)}")
print()

# Test case 5: range(0, 10, 1) - normal valid range for comparison
r = range(0, 10, 1)
print(f"Test 5: range(0, 10, 1) [normal range for comparison]")
print(f"  Python len(range(0, 10, 1)) = {len(r)}")
print(f"  pandas length_of_indexer(range(0, 10, 1)) = {indexers.length_of_indexer(r)}")