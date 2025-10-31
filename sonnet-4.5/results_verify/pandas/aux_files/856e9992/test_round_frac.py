import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import numpy as np

def _round_frac(x, precision):
    """Round a number to the given precision"""
    if not np.isfinite(x) or x == 0:
        return x
    frac, whole = np.modf(x)
    if whole == 0:
        digits = -int(np.floor(np.log10(np.abs(x)))) - 1 + precision
    else:
        digits = precision
    return np.around(x, digits)

# Test with the problematic value
test_values = [0.0, 5e-324, 1e-323, 1e-320, 1e-310, 1e-300, 0.001]

for val in test_values:
    try:
        result = _round_frac(val, precision=3)
        print(f"_round_frac({val}, 3) = {result}")
    except Exception as e:
        print(f"_round_frac({val}, 3) raised error: {e}")

# The issue is likely in np.log10 of very small numbers
print("\n" + "="*60)
print("Testing np.log10 with very small numbers:")
for val in [5e-324, 1e-323, 1e-320]:
    log_val = np.log10(val) if val != 0 else "N/A"
    print(f"np.log10({val}) = {log_val}")
    if val != 0:
        floor_log = np.floor(log_val)
        print(f"  np.floor(np.log10({val})) = {floor_log}")
        print(f"  -int(floor) - 1 + 3 = {-int(floor_log) - 1 + 3}")