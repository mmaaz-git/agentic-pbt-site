from dask.core import istask

# Test various expressions
exprs = [
    (dict, []),                          # Empty dict
    (dict, [['a', 1], ['b', 2]]),       # Non-empty dict
    (list, [1, 2, 3]),                   # List
    (tuple, [1, 2, 3]),                  # Tuple
    (set, [1, 2, 3]),                    # Set
    [1, 2, 3],                           # Plain list
    {'a': 1},                            # Plain dict
]

for expr in exprs:
    print(f"istask({expr!r}): {istask(expr)}")