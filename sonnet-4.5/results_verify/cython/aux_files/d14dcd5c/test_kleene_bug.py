import numpy as np
import sys

# Test basic reproduction
try:
    from pandas.core.ops import kleene_and, kleene_or, kleene_xor

    left = np.array([True, False])
    right = np.array([True, True])

    print("Testing kleene_and with both masks None...")
    sys.setrecursionlimit(100)  # Set a smaller limit to catch recursion quickly
    try:
        result = kleene_and(left, right, None, None)
        print(f"kleene_and succeeded: result={result}")
    except RecursionError as e:
        print(f"kleene_and failed with RecursionError: {e}")

    print("\nTesting kleene_or with both masks None...")
    try:
        result = kleene_or(left, right, None, None)
        print(f"kleene_or succeeded: result={result}")
    except RecursionError as e:
        print(f"kleene_or failed with RecursionError: {e}")

    print("\nTesting kleene_xor with both masks None...")
    try:
        result = kleene_xor(left, right, None, None)
        print(f"kleene_xor succeeded: result={result}")
    except RecursionError as e:
        print(f"kleene_xor failed with RecursionError: {e}")

except ImportError as e:
    print(f"Could not import pandas functions: {e}")