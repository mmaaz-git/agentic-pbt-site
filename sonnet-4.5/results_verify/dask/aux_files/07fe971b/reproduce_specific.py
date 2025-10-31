size = 6356
blocksize = 37

num_blocks = size // blocksize  # 171 blocks
blocksize_float = size / num_blocks  # 37.16959064327485 (WRONG: uses float division)
blocksize_int = size // num_blocks   # 37 (CORRECT: uses integer division)

print(f"num_blocks: {num_blocks}")
print(f"blocksize_float: {blocksize_float}")
print(f"blocksize_int: {blocksize_int}")
print(f"Difference in blocksize: {blocksize_float - blocksize_int}")
print()

place_float = 0.0
offsets_float = [0]
while size - place_float > (blocksize_float * 2) - 1:
    place_float += blocksize_float
    offsets_float.append(int(place_float))

place_int = 0
offsets_int = [0]
while size - place_int > (blocksize_int * 2) - 1:
    place_int += blocksize_int
    offsets_int.append(place_int)

print(f"Total offsets with float: {len(offsets_float)}")
print(f"Total offsets with int: {len(offsets_int)}")
print()

if len(offsets_float) > 170 and len(offsets_int) > 170:
    print(f"Float-based offset 170: {offsets_float[170]}")  # 6318
    print(f"Int-based offset 170: {offsets_int[170]}")      # 6290
    print(f"Difference: {offsets_float[170] - offsets_int[170]} bytes")  # 28 bytes off!
    print()

# Show some differences
print("First 10 differences in offsets:")
for i in range(min(len(offsets_float), len(offsets_int))):
    if offsets_float[i] != offsets_int[i]:
        print(f"  Index {i}: float={offsets_float[i]}, int={offsets_int[i]}, diff={offsets_float[i] - offsets_int[i]}")