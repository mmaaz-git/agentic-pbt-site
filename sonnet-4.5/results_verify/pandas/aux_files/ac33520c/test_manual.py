import pandas.core.indexers as indexers

r = range(1, 0, 1)
print(f"len(range(1, 0, 1)) = {len(r)}")
print(f"length_of_indexer(range(1, 0, 1)) = {indexers.length_of_indexer(r)}")

r = range(10, 0, 2)
print(f"len(range(10, 0, 2)) = {len(r)}")
print(f"length_of_indexer(range(10, 0, 2)) = {indexers.length_of_indexer(r)}")

# Let's test a few more cases
print("\nAdditional test cases:")
test_cases = [
    (5, 5, 1),  # start == stop
    (0, 5, 1),  # normal positive case
    (10, 5, 1), # start > stop with positive step
    (100, 0, 3) # larger values
]

for start, stop, step in test_cases:
    r = range(start, stop, step)
    print(f"range({start}, {stop}, {step}): len() = {len(r)}, length_of_indexer() = {indexers.length_of_indexer(r)}")