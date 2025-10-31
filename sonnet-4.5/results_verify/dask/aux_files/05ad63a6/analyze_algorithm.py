#!/usr/bin/env python3

def analyze_loop(size, blocksize):
    """Analyze the loop behavior with different parameters"""
    if size % blocksize and size > blocksize:
        blocksize1 = size / (size // blocksize)
    else:
        blocksize1 = blocksize

    place = 0
    iterations = 0
    offsets = [0]

    while size - place > (blocksize1 * 2) - 1:
        iterations += 1
        place += blocksize1
        offsets.append(int(place))

    return iterations, blocksize1, offsets

# Test various cases
test_cases = [
    (1000, 100),   # Normal case
    (1000, 1),     # Small blocksize
    (1_000_000, 1),  # Large file, tiny blocksize
    (1000, 999),   # Blocksize almost as big as file
    (141335, 1),   # The specific failing case from report
]

for size, blocksize in test_cases:
    iters, blocksize1, offsets = analyze_loop(size, blocksize)
    print(f"\nsize={size:,}, blocksize={blocksize}")
    print(f"  blocksize1={blocksize1:.2f}")
    print(f"  iterations={iters:,}")
    print(f"  num_offsets={len(offsets)}")
    print(f"  Time complexity: O({iters}/n) where n is file size")

    # Show what the algorithm is actually doing
    if len(offsets) <= 10:
        print(f"  offsets={offsets}")
    else:
        print(f"  offsets (first 5)={offsets[:5]}")
        print(f"  offsets (last 5)={offsets[-5:]}")

# Analyze the mathematical relationship
print("\n\n=== Mathematical Analysis ===")
print("When blocksize is small relative to size:")
print("- blocksize1 = size / (size // blocksize) ≈ blocksize (for small blocksize)")
print("- Loop condition: size - place > (blocksize1 * 2) - 1")
print("- Each iteration: place += blocksize1")
print("- Number of iterations ≈ size / blocksize1 ≈ size / blocksize")
print("\nThis gives O(size/blocksize) time complexity!")
print("\nFor blocksize=1 and size=1GB:")
print(f"  Iterations needed: {1_000_000_000 // 1:,}")
print("  This is clearly impractical!")