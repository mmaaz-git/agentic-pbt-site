import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env')

from scipy.optimize import brentq

# Testing the same polynomial function x^3
try:
    result = brentq(lambda x: x**3, -1.0, 2.0, xtol=1e-6, rtol=1e-6, maxiter=10, full_output=True)
    print(f"brentq result: {result}")
except RuntimeError as e:
    print(f"brentq raised RuntimeError: {e}")
except Exception as e:
    print(f"brentq raised unexpected exception: {type(e).__name__}: {e}")