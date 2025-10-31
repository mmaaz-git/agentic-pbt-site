#!/usr/bin/env python3
"""Understanding what istask considers a task"""

from dask.core import istask

# Test what istask considers a task
test_cases = [
    ("dict with empty list", (dict, [])),
    ("dict with list of pairs", (dict, [['a', 1], ['b', 2]])),
    ("tuple constructor", (tuple, [1, 2, 3])),
    ("list constructor", (list, [1, 2, 3])),
    ("set constructor", (set, [1, 2, 3])),
    ("regular tuple", (1, 2, 3)),
    ("function with args", (lambda x: x + 1, 5)),
    ("string", "hello"),
    ("number", 42),
    ("empty tuple", ()),
    ("single element tuple", (42,)),
]

print("Testing what istask considers a 'task':\n")
for description, value in test_cases:
    result = istask(value)
    print(f"{description:30} {str(value):40} -> istask: {result}")