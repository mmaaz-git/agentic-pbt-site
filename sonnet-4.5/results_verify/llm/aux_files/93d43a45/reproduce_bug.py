import sys
sys.path.insert(0, "/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages")

import llm

# Test the specific failing case from the bug report
try:
    result = llm.cosine_similarity([2.225073858507203e-309], [2.225073858507203e-309])
    print(f"Result: {result}")
except ZeroDivisionError as e:
    print(f"ZeroDivisionError: {e}")

# Let's also check what happens with the magnitude calculation
a = [2.225073858507203e-309]
squared = [x * x for x in a]
print(f"Original value: {a[0]}")
print(f"Squared value: {squared[0]}")
print(f"Sum of squares: {sum(squared)}")
print(f"Magnitude: {sum(squared) ** 0.5}")

# Check if underflow is happening
print(f"\nIs squared value exactly 0? {squared[0] == 0.0}")
print(f"Is magnitude exactly 0? {sum(squared) ** 0.5 == 0.0}")