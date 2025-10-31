import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/attrs_env/lib/python3.13/site-packages')

import operator
from attrs import validators

gt_validator = validators.gt(10)

print("Docstring says: 'The validator uses `operator.ge`'")
print(f"Actual implementation: {gt_validator.compare_func}")
print(f"operator.gt: {operator.gt}")
print(f"operator.ge: {operator.ge}")
print(f"Uses operator.gt: {gt_validator.compare_func == operator.gt}")
print(f"Uses operator.ge: {gt_validator.compare_func == operator.ge}")