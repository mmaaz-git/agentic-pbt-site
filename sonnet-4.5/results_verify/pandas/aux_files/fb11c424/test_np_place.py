import numpy as np

print("Understanding the difference between np.putmask and np.place\n")

# Testing with exact match
print("1. EXACT MATCH (10 mask positions, 10 values):")
values1a = np.arange(10)
values1b = np.arange(10)
mask1 = np.ones(10, dtype=bool)
new1 = np.arange(100, 110)

print(f"  mask sum: {mask1.sum()}, new length: {len(new1)}")
np.putmask(values1a, mask1, new1)
np.place(values1b, mask1, new1)
print(f"  np.putmask result: {values1a}")
print(f"  np.place result:   {values1b}")
print(f"  Results identical: {np.array_equal(values1a, values1b)}")

# Testing with fewer values than mask positions
print("\n2. FEWER VALUES (10 mask positions, 3 values):")
values2a = np.arange(10)
values2b = np.arange(10)
mask2 = np.ones(10, dtype=bool)
new2 = np.array([100, 200, 300])

print(f"  mask sum: {mask2.sum()}, new length: {len(new2)}")
np.putmask(values2a, mask2, new2)
np.place(values2b, mask2, new2)
print(f"  np.putmask result: {values2a}")
print(f"  np.place result:   {values2b}")
print(f"  Results identical: {np.array_equal(values2a, values2b)}")

# Testing with single value
print("\n3. SINGLE VALUE (10 mask positions, 1 value):")
values3a = np.arange(10)
values3b = np.arange(10)
mask3 = np.ones(10, dtype=bool)
new3 = np.array([999])

print(f"  mask sum: {mask3.sum()}, new length: {len(new3)}")
np.putmask(values3a, mask3, new3)
np.place(values3b, mask3, new3)
print(f"  np.putmask result: {values3a}")
print(f"  np.place result:   {values3b}")
print(f"  Results identical: {np.array_equal(values3a, values3b)}")

# The key difference according to comment in the code
print("\nKEY INSIGHT from pandas source code comment:")
print("Line 86-89 of putmask.py says:")
print("# If length of ``new`` is less than the length of ``values``,")
print("# `np.putmask` would first repeat the ``new`` array and then")
print("# assign the masked values hence produces incorrect result.")
print("# `np.place` on the other hand uses the ``new`` values at it is")

print("\nBUT in practice, both np.putmask and np.place repeat values!")
print("So the comment appears to be incorrect or outdated.")