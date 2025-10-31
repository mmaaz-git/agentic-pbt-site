import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.utils import schema_dsl

# Test case with empty field name before colon
result = schema_dsl(" : description")
print("Result:", result)