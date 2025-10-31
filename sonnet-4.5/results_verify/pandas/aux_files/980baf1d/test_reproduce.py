#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from pandas.compat.numpy.function import ARGSORT_DEFAULTS, validate_argsort

print("ARGSORT_DEFAULTS:", ARGSORT_DEFAULTS)

try:
    validate_argsort((), {"kind": "quicksort"})
    print("No error raised for kind='quicksort'")
except ValueError as e:
    print(f"ValueError raised: {e}")