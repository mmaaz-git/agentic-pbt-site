import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env')

from scipy.optimize.cython_optimize import _zeros

args = (0.0, 0.0, 0.0, 1.0)
xa, xb = -1.0, 2.0
xtol, rtol, mitr = 1e-6, 1e-6, 10

output = _zeros.full_output_example(args, xa, xb, xtol, rtol, mitr)

print(f"error_num: {output['error_num']}")
print(f"Output: {output}")

assert output['error_num'] >= 0, f"error_num is negative: {output['error_num']}"