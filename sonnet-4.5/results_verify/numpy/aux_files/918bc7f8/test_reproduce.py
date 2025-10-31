import numpy as np
import numpy.strings as nps

test_cases = [
    ('\x00', 3),
    ('\x00\x00', 2),
    ('a\x00', 2),
]

for s, n in test_cases:
    arr = np.array([s], dtype=str)
    result = nps.multiply(arr, n)[0]
    expected = s * n
    print(f"multiply({repr(s):10}, {n}): Expected={repr(expected):20}, Got={repr(result):20}")