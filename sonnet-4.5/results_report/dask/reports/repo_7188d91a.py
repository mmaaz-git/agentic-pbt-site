import dask.bag as db

# Test case 1: Basic case from the bug report
seq = [1, 2, 3, 4]
requested_npartitions = 3

bag = db.from_sequence(seq, npartitions=requested_npartitions)

print(f"Test 1: Basic case")
print(f"Sequence: {seq}")
print(f"Requested npartitions: {requested_npartitions}")
print(f"Actual npartitions: {bag.npartitions}")
print(f"Expected: 3, Got: {bag.npartitions}")
print()

# Test case 2: Specific failing case from bug report
seq2 = [0, 0, 0, 0, 0]
requested_npartitions2 = 4

bag2 = db.from_sequence(seq2, npartitions=requested_npartitions2)

print(f"Test 2: Specific failing case")
print(f"Sequence: {seq2}")
print(f"Requested npartitions: {requested_npartitions2}")
print(f"Actual npartitions: {bag2.npartitions}")
print(f"Expected: 4, Got: {bag2.npartitions}")
print()

# Test case 3: Downstream zip() problem
print("Test 3: Downstream zip() problem")
bag1 = db.from_sequence([0, 0, 0], npartitions=3)
bag2 = db.from_sequence([0, 0, 0, 0], npartitions=3)
print(f"bag1 ([0, 0, 0], npartitions=3): {bag1.npartitions} partitions")
print(f"bag2 ([0, 0, 0, 0], npartitions=3): {bag2.npartitions} partitions")

try:
    result = db.zip(bag1, bag2)
    print("zip() succeeded")
except AssertionError as e:
    print(f"zip() failed with AssertionError: {e}")

# Final assertion to demonstrate the bug
assert bag.npartitions == requested_npartitions, f"Expected {requested_npartitions} partitions, got {bag.npartitions}"