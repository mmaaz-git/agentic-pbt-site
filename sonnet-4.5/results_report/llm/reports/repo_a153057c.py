import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.utils import truncate_string

text = "hello"
max_length = 1
result = truncate_string(text, max_length)

print(f"Input: '{text}' (length {len(text)})")
print(f"Max length: {max_length}")
print(f"Result: '{result}' (length {len(result)})")
print(f"Constraint violated: {len(result)} > {max_length}")