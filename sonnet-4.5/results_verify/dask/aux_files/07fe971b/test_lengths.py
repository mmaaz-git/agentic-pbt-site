"""Check how the lengths are calculated with float vs int division"""

size = 6356
blocksize = 37

# Float version (current implementation)
num_blocks = size // blocksize
blocksize_float = size / num_blocks

place_float = 0.0
offsets_float = [0]
while size - place_float > (blocksize_float * 2) - 1:
    place_float += blocksize_float
    offsets_float.append(int(place_float))

# Calculate lengths from offsets (as done in the actual code)
lengths_float = []
for i in range(1, len(offsets_float)):
    lengths_float.append(offsets_float[i] - offsets_float[i-1])
lengths_float.append(size - offsets_float[-1])

# Int version (proposed fix)
blocksize_int = size // num_blocks

place_int = 0
offsets_int = [0]
while size - place_int > (blocksize_int * 2) - 1:
    place_int += blocksize_int
    offsets_int.append(place_int)

# Calculate lengths from offsets
lengths_int = []
for i in range(1, len(offsets_int)):
    lengths_int.append(offsets_int[i] - offsets_int[i-1])
lengths_int.append(size - offsets_int[-1])

print("Float version:")
print(f"  Offsets: {offsets_float[:5]}...{offsets_float[-3:]}")
print(f"  Lengths: {lengths_float[:5]}...{lengths_float[-3:]}")
print(f"  Total bytes: {sum(lengths_float)}")
print(f"  All lengths equal? {len(set(lengths_float[:-1])) == 1}")
print()

print("Int version:")
print(f"  Offsets: {offsets_int[:5]}...{offsets_int[-3:]}")
print(f"  Lengths: {lengths_int[:5]}...{lengths_int[-3:]}")
print(f"  Total bytes: {sum(lengths_int)}")
print(f"  All lengths equal? {len(set(lengths_int[:-1])) == 1}")
print()

print("Differences:")
print(f"  Different offsets? {offsets_float != offsets_int}")
print(f"  Different lengths? {lengths_float != lengths_int}")
print(f"  Both sum to correct size? {sum(lengths_float) == size and sum(lengths_int) == size}")