import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

import llm

a = [1, 0]
b = [1]
result = llm.cosine_similarity(a, b)

print(f"cosine_similarity({a}, {b}) = {result}")

# Let's also trace through the calculation
import math

# What zip does - it truncates to the shorter
pairs = list(zip(a, b))
print(f"zip(a, b) = {pairs}")

dot_product = sum(x * y for x, y in zip(a, b))
print(f"dot_product = {dot_product}")

magnitude_a = sum(x * x for x in a) ** 0.5
magnitude_b = sum(x * x for x in b) ** 0.5
print(f"magnitude_a = {magnitude_a}")
print(f"magnitude_b = {magnitude_b}")

computed_result = dot_product / (magnitude_a * magnitude_b)
print(f"computed result = {computed_result}")

# What the correct result should be
# Cosine similarity is UNDEFINED for vectors of different dimensions