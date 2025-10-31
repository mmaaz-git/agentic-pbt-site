import numpy as np
import pandas as pd
from pandas.core.arrays.sparse import SparseArray
from random import randint

def test_sparse_array_astype_preserves_values(data, fill_value1, fill_value2):
    """Test that astype preserves values when changing fill_value"""
    try:
        arr = np.array(data)
        sparse = SparseArray(arr, fill_value=fill_value1)

        dtype = pd.SparseDtype(np.float64, fill_value2)
        sparse_casted = sparse.astype(dtype)

        expected = arr.astype(np.float64)
        actual = sparse_casted.to_dense()

        # Check if arrays are equal
        return np.allclose(actual, expected)
    except Exception as e:
        print(f"Exception: {e}")
        return False

print("Testing property: astype should preserve values when changing fill_value")
print("=" * 70)

# Run tests manually
failure_count = 0
success_count = 0
failures = []

test_cases = []

# Add specific edge cases
test_cases.append(([0], 0, 1))  # The bug report case
test_cases.append(([5, 5, 5], 5, 10))  # All values equal fill_value
test_cases.append(([-1, -1], -1, 0))  # Negative fill_value
test_cases.append(([100], 100, -100))  # Single value

# Add random cases
for _ in range(20):
    size = randint(1, 10)
    data = [randint(-20, 20) for _ in range(size)]
    fill_value1 = randint(-20, 20)
    fill_value2 = randint(-20, 20)
    test_cases.append((data, fill_value1, fill_value2))

# Run all test cases
for data, fill_value1, fill_value2 in test_cases:
    if test_sparse_array_astype_preserves_values(data, fill_value1, fill_value2):
        success_count += 1
    else:
        failure_count += 1
        # Check if this is the specific pattern: all values equal fill_value1
        if all(v == fill_value1 for v in data):
            failures.append({
                'data': data,
                'fill_value1': fill_value1,
                'fill_value2': fill_value2,
                'pattern': 'all_equal_fill'
            })
        else:
            failures.append({
                'data': data,
                'fill_value1': fill_value1,
                'fill_value2': fill_value2,
                'pattern': 'mixed'
            })

print(f"\nTest Summary:")
print(f"  Total tests: {len(test_cases)}")
print(f"  Successes: {success_count}")
print(f"  Failures: {failure_count}")
print(f"  Failure rate: {failure_count/len(test_cases)*100:.1f}%")

if failures:
    print(f"\nFailure patterns:")
    all_equal_fill = [f for f in failures if f['pattern'] == 'all_equal_fill']
    mixed = [f for f in failures if f['pattern'] == 'mixed']

    print(f"  All values equal to fill_value: {len(all_equal_fill)} failures")
    print(f"  Mixed values: {len(mixed)} failures")

    if all_equal_fill:
        print(f"\nFirst few 'all_equal_fill' failures:")
        for f in all_equal_fill[:3]:
            print(f"    data={f['data']}, fill {f['fill_value1']}->{f['fill_value2']}")

print("\n" + "=" * 70)
print("Detailed test of the reported bug case:")
data = [0]
sparse = SparseArray(data, fill_value=0)
dtype = pd.SparseDtype(np.float64, fill_value=1)
result = sparse.astype(dtype).to_dense()
expected = np.array(data).astype(np.float64)

print(f"Data: {data}")
print(f"Fill value: 0 -> 1")
print(f"Expected result: {expected}")
print(f"Actual result: {result}")
print(f"Bug confirmed: {not np.array_equal(result, expected)}")

# Check when the bug occurs
print("\n" + "=" * 70)
print("Testing when the bug occurs:")

# Test case 1: All values equal fill_value -> bug occurs
data = [5, 5, 5]
sparse = SparseArray(data, fill_value=5)
dtype = pd.SparseDtype(np.float64, fill_value=10)
result = sparse.astype(dtype).to_dense()
expected = np.array(data).astype(np.float64)
print(f"\nCase 1: All values equal fill_value")
print(f"  Data: {data}, fill_value: 5 -> 10")
print(f"  Expected: {expected}, Actual: {result}")
print(f"  Bug: {not np.array_equal(result, expected)}")

# Test case 2: Some values equal fill_value -> check behavior
data = [5, 6, 5]
sparse = SparseArray(data, fill_value=5)
dtype = pd.SparseDtype(np.float64, fill_value=10)
result = sparse.astype(dtype).to_dense()
expected = np.array(data).astype(np.float64)
print(f"\nCase 2: Some values equal fill_value")
print(f"  Data: {data}, fill_value: 5 -> 10")
print(f"  Expected: {expected}, Actual: {result}")
print(f"  Bug: {not np.array_equal(result, expected)}")

# Test case 3: No values equal fill_value -> should work
data = [6, 7, 8]
sparse = SparseArray(data, fill_value=5)
dtype = pd.SparseDtype(np.float64, fill_value=10)
result = sparse.astype(dtype).to_dense()
expected = np.array(data).astype(np.float64)
print(f"\nCase 3: No values equal fill_value")
print(f"  Data: {data}, fill_value: 5 -> 10")
print(f"  Expected: {expected}, Actual: {result}")
print(f"  Bug: {not np.array_equal(result, expected)}")

# Test case 4: fill_value doesn't change -> should always work
data = [5, 5, 5]
sparse = SparseArray(data, fill_value=5)
dtype = pd.SparseDtype(np.float64, fill_value=5)
result = sparse.astype(dtype).to_dense()
expected = np.array(data).astype(np.float64)
print(f"\nCase 4: fill_value doesn't change")
print(f"  Data: {data}, fill_value: 5 -> 5")
print(f"  Expected: {expected}, Actual: {result}")
print(f"  Bug: {not np.array_equal(result, expected)}")