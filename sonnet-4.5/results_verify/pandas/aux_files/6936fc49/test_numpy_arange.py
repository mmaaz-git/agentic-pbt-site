import numpy as np

print("Testing numpy.arange with step=0:")
try:
    result = np.arange(0, 10, 0)
    print(f"Result: {result}")
except ZeroDivisionError as e:
    print(f"ZeroDivisionError: {e}")
except Exception as e:
    print(f"Other error - {type(e).__name__}: {e}")