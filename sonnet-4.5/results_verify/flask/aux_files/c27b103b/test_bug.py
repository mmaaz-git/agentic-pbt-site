#!/usr/bin/env python3
"""Test script to reproduce the Flask environment variable whitespace bug."""

import os
import sys

# Add Flask environment to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/flask_env/lib/python3.13/site-packages')

from flask.helpers import get_debug_flag, get_load_dotenv

# Test get_debug_flag()
print("Testing get_debug_flag():")
print("-" * 40)

os.environ['FLASK_DEBUG'] = 'false'
result1 = get_debug_flag()
print(f"FLASK_DEBUG='false' -> {result1}")

os.environ['FLASK_DEBUG'] = ' false'
result2 = get_debug_flag()
print(f"FLASK_DEBUG=' false' -> {result2}")

os.environ['FLASK_DEBUG'] = 'false '
result3 = get_debug_flag()
print(f"FLASK_DEBUG='false ' -> {result3}")

os.environ['FLASK_DEBUG'] = '\tfalse'
result4 = get_debug_flag()
print(f"FLASK_DEBUG='\\tfalse' -> {result4}")

print(f"\nExpected: All should return False")
print(f"Actual: 'false'={result1}, ' false'={result2}, 'false '={result3}, '\\tfalse'={result4}")

# Test get_load_dotenv()
print("\n" + "=" * 40)
print("Testing get_load_dotenv():")
print("-" * 40)

os.environ['FLASK_SKIP_DOTENV'] = 'false'
result5 = get_load_dotenv(True)
print(f"FLASK_SKIP_DOTENV='false' -> {result5}")

os.environ['FLASK_SKIP_DOTENV'] = ' false'
result6 = get_load_dotenv(True)
print(f"FLASK_SKIP_DOTENV=' false' -> {result6}")

os.environ['FLASK_SKIP_DOTENV'] = 'false '
result7 = get_load_dotenv(True)
print(f"FLASK_SKIP_DOTENV='false ' -> {result7}")

os.environ['FLASK_SKIP_DOTENV'] = '\tfalse'
result8 = get_load_dotenv(True)
print(f"FLASK_SKIP_DOTENV='\\tfalse' -> {result8}")

print(f"\nExpected: All should return True")
print(f"Actual: 'false'={result5}, ' false'={result6}, 'false '={result7}, '\\tfalse'={result8}")

# Cleanup
if 'FLASK_DEBUG' in os.environ:
    del os.environ['FLASK_DEBUG']
if 'FLASK_SKIP_DOTENV' in os.environ:
    del os.environ['FLASK_SKIP_DOTENV']