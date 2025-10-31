import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env')

from scipy.optimize.cython_optimize import _zeros
import inspect

# Get the docstring
print("Docstring for full_output_example:")
print("=" * 50)
print(_zeros.full_output_example.__doc__)
print("=" * 50)

# Get function signature
print("\nFunction signature:")
print(inspect.signature(_zeros.full_output_example))