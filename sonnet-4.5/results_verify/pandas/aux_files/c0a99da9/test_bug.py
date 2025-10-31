import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from pandas.compat.numpy.function import ARGSORT_DEFAULTS

print("ARGSORT_DEFAULTS contents:")
for key, value in ARGSORT_DEFAULTS.items():
    print(f"  {key}: {value!r}")

print(f"\nBug: 'kind' is {ARGSORT_DEFAULTS['kind']!r}, expected 'quicksort'")