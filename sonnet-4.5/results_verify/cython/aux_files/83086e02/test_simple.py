import os
import tempfile
from Cython.Build.Dependencies import extended_iglob

with tempfile.TemporaryDirectory() as tmpdir:
    filepath = os.path.join(tmpdir, "test.txt")
    with open(filepath, 'w') as f:
        f.write('')

    pattern = os.path.join(tmpdir, '{test,test}.txt')
    results = list(extended_iglob(pattern))

    print(f"Pattern: {pattern}")
    print(f"Results: {results}")
    print(f"Number of results: {len(results)}")
    print(f"Unique results: {len(set(results))}")

    assert len(results) == len(set(results)), f"Expected unique results, got duplicates: {results}"