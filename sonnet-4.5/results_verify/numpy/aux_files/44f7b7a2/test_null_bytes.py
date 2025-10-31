import numpy as np
import numpy.strings as ns

# Test with manual input
strings = ['\x00']
arr = np.array(strings, dtype=np.str_)
result = ns.upper(arr)

for orig, res in zip(strings, result):
    expected = orig.upper()
    print(f"Input: {repr(orig)}")
    print(f"Expected (Python): {repr(expected)}")
    print(f"Got (NumPy): {repr(res)}")
    try:
        assert res == expected
        print("Test PASSED")
    except AssertionError:
        print("Test FAILED - NumPy result differs from Python!")