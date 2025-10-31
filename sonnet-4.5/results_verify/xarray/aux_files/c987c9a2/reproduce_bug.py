import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages')

from pydantic.experimental.pipeline import _apply_constraint
import annotated_types

with open('/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages/pydantic/experimental/pipeline.py', 'r') as f:
    lines = f.readlines()

print("Gt constraint (lines 448-463) - HAS else clause:")
print(''.join(lines[447:463]))

print("\nGe constraint (lines 464-478) - MISSING else clause:")
print(''.join(lines[463:479]))

print("\nNotice: Gt has 'else:' before check_gt definition")
print("Notice: Ge does NOT have 'else:', so check_ge always runs")