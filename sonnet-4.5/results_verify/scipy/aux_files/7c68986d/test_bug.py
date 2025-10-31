import tempfile
import os
import numpy as np
from scipy.io import mmread, mmwrite
from scipy.sparse import coo_array

print("=== Testing empty complex sparse matrix ===")

# Test 1: Empty complex sparse matrix
data = np.array([[0.0 + 0.0j]])
sparse_matrix = coo_array(data)

print(f"Original dtype: {sparse_matrix.dtype}")
print(f"Original nnz: {sparse_matrix.nnz}")
print(f"Original data: {sparse_matrix.data}")
print(f"Original data dtype: {sparse_matrix.data.dtype}")

with tempfile.NamedTemporaryFile(suffix='.mtx', delete=False, mode='w') as f:
    filename = f.name

try:
    mmwrite(filename, sparse_matrix)

    with open(filename, 'r') as f:
        content = f.read()
        print(f"\nFile content:")
        print(content)

    result = mmread(filename, spmatrix=False)

    print(f"\nResult dtype: {result.dtype}")
    print(f"Result nnz: {result.nnz}")
    print(f"Dtypes match: {result.dtype == sparse_matrix.dtype}")

finally:
    if os.path.exists(filename):
        os.remove(filename)

print("\n=== Testing non-empty complex sparse matrix ===")

# Test 2: Non-empty complex sparse matrix
data2 = np.array([[1.0 + 2.0j]])
sparse_matrix2 = coo_array(data2)

print(f"Original dtype: {sparse_matrix2.dtype}")
print(f"Original nnz: {sparse_matrix2.nnz}")
print(f"Original data: {sparse_matrix2.data}")

with tempfile.NamedTemporaryFile(suffix='.mtx', delete=False, mode='w') as f:
    filename2 = f.name

try:
    mmwrite(filename2, sparse_matrix2)

    with open(filename2, 'r') as f:
        content = f.read()
        print(f"\nFile content:")
        print(content)

    result2 = mmread(filename2, spmatrix=False)

    print(f"\nResult dtype: {result2.dtype}")
    print(f"Result nnz: {result2.nnz}")
    print(f"Dtypes match: {result2.dtype == sparse_matrix2.dtype}")

finally:
    if os.path.exists(filename2):
        os.remove(filename2)

print("\n=== Testing empty real sparse matrix ===")

# Test 3: Empty real sparse matrix
data3 = np.array([[0.0]])
sparse_matrix3 = coo_array(data3)

print(f"Original dtype: {sparse_matrix3.dtype}")
print(f"Original nnz: {sparse_matrix3.nnz}")

with tempfile.NamedTemporaryFile(suffix='.mtx', delete=False, mode='w') as f:
    filename3 = f.name

try:
    mmwrite(filename3, sparse_matrix3)

    with open(filename3, 'r') as f:
        header = f.readline()
        print(f"File header: {header.strip()}")

    result3 = mmread(filename3, spmatrix=False)

    print(f"Result dtype: {result3.dtype}")
    print(f"Dtypes match: {result3.dtype == sparse_matrix3.dtype}")

finally:
    if os.path.exists(filename3):
        os.remove(filename3)