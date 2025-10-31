import numpy as np
from pandas.arrays import SparseArray

data = [0]
fill_value = 0

sparse = SparseArray(data, fill_value=fill_value)
dense = np.array(data)

print("Dense array argmin result:", dense.argmin())
print("Dense array argmax result:", dense.argmax())

print("\nAttempting sparse array argmin...")
try:
    result = sparse.argmin()
    print("Sparse array argmin result:", result)
except Exception as e:
    print(f"Error in sparse.argmin(): {type(e).__name__}: {e}")

print("\nAttempting sparse array argmax...")
try:
    result = sparse.argmax()
    print("Sparse array argmax result:", result)
except Exception as e:
    print(f"Error in sparse.argmax(): {type(e).__name__}: {e}")