import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from pandas.compat.numpy.function import validate_argsort

try:
    validate_argsort((), {"kind": "quicksort"})
    print("Validation passed")
except ValueError as e:
    print(f"ValueError raised: {e}")