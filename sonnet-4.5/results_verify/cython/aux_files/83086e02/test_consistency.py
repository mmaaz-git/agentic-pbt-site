import os
import tempfile
from Cython.Build.Dependencies import extended_iglob

# Test consistency between different code paths

with tempfile.TemporaryDirectory() as tmpdir:
    # Create test files
    subdir = os.path.join(tmpdir, "sub")
    os.makedirs(subdir)

    with open(os.path.join(tmpdir, "a.txt"), 'w') as f:
        f.write('')
    with open(os.path.join(subdir, "a.txt"), 'w') as f:
        f.write('')

    # Test brace expansion with duplicates
    pattern1 = os.path.join(tmpdir, '{a,a}.txt')
    results1 = list(extended_iglob(pattern1))
    print(f"Brace expansion pattern: {pattern1}")
    print(f"Results: {results1}")
    print(f"Number of results: {len(results1)}, Unique: {len(set(results1))}")
    print()

    # Test ** pattern (recursive search) with a pattern that would find duplicates
    # by exploring the same path twice
    pattern2 = os.path.join(tmpdir, '**/a.txt')
    results2 = list(extended_iglob(pattern2))
    print(f"Recursive pattern: {pattern2}")
    print(f"Results: {results2}")
    print(f"Number of results: {len(results2)}, Unique: {len(set(results2))}")
    print()

    # Test if the ** code path prevents duplicates
    pattern3 = os.path.join(tmpdir, '**', 'a.txt')
    results3 = list(extended_iglob(pattern3))
    print(f"Recursive pattern (direct): {pattern3}")
    print(f"Results: {results3}")
    print(f"Number of results: {len(results3)}, Unique: {len(set(results3))}")