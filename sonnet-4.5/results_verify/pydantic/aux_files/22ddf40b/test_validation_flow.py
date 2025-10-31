import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages')

import pydantic_core
from pydantic.experimental import pipeline
from typing import Any
import annotated_types

# Let's trace the actual execution flow
def trace_apply_constraint():
    # Create an integer schema
    s = {'type': 'int'}

    print("Testing Gt constraint (CORRECT - has else clause):")
    constraint_gt = annotated_types.Gt(10)
    result_gt = pipeline._apply_constraint(s, constraint_gt)
    print(f"  Result schema: {result_gt}")
    print(f"  Has 'gt' key in schema: {'gt' in result_gt}")
    print(f"  Has 'function' key (validator): {'function' in result_gt}")

    print("\nTesting Ge constraint (BUG - missing else clause):")
    constraint_ge = annotated_types.Ge(10)
    result_ge = pipeline._apply_constraint(s, constraint_ge)
    print(f"  Result schema: {result_ge}")
    print(f"  Has 'ge' key in schema: {'ge' in result_ge}")
    print(f"  Has 'function' key (validator): {'function' in result_ge}")

    print("\nTesting Lt constraint (BUG - missing else clause):")
    constraint_lt = annotated_types.Lt(100)
    result_lt = pipeline._apply_constraint(s, constraint_lt)
    print(f"  Result schema: {result_lt}")
    print(f"  Has 'lt' key in schema: {'lt' in result_lt}")
    print(f"  Has 'function' key (validator): {'function' in result_lt}")

    print("\nTesting Le constraint (BUG - missing else clause):")
    constraint_le = annotated_types.Le(100)
    result_le = pipeline._apply_constraint(s, constraint_le)
    print(f"  Result schema: {result_le}")
    print(f"  Has 'le' key in schema: {'le' in result_le}")
    print(f"  Has 'function' key (validator): {'function' in result_le}")

    print("\nTesting MultipleOf constraint (BUG - missing else clause):")
    constraint_multiple = annotated_types.MultipleOf(5)
    result_multiple = pipeline._apply_constraint(s, constraint_multiple)
    print(f"  Result schema: {result_multiple}")
    print(f"  Has 'multiple_of' key in schema: {'multiple_of' in result_multiple}")
    print(f"  Has 'function' key (validator): {'function' in result_multiple}")

trace_apply_constraint()

print("\n" + "="*60)
print("Analysis:")
print("- Gt: Only sets schema constraint OR validator (correct)")
print("- Ge/Lt/Le/MultipleOf: Sets BOTH schema constraint AND validator (double validation)")
print("="*60)