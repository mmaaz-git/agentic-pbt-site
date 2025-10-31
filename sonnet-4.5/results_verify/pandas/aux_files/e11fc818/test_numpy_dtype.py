import numpy as np

# Test how np.dtype handles the problematic inputs
test_inputs = ['0:', '1:', '0;', '0/', 'invalid', 'foo:bar']

for test_input in test_inputs:
    print(f"Testing np.dtype({test_input!r}):")
    try:
        result = np.dtype(test_input)
        print(f"  Success: {result}")
    except TypeError as e:
        print(f"  TypeError: {e}")
    except ValueError as e:
        print(f"  ValueError: {e}")
    except Exception as e:
        print(f"  Other exception ({type(e).__name__}): {e}")
    print()