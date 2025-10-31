from Cython.Utility import pylong_join, _pylong_join

test_cases = [-10, -1, 0, 1, 2, 3]

for count in test_cases:
    result1 = pylong_join(count)
    result2 = _pylong_join(count)
    match = "✓" if result1 == result2 else "✗"
    print(f"count={count:3}: {match} pylong_join={result1!r:40} _pylong_join={result2!r}")