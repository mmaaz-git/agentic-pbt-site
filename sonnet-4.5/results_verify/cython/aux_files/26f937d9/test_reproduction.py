#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.utils import schema_dsl

# Test case from bug report
schema_str = "field1 str, , field2 int"

try:
    result = schema_dsl(schema_str)
    print("Result:", result)
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    import traceback
    traceback.print_exc()