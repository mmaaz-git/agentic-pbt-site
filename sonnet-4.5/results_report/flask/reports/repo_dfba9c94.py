#!/usr/bin/env python3
"""Minimal reproduction of Flask Config.from_prefixed_env type collision bug"""
import os
import tempfile
from flask import Config

# Clean any existing FLASK_ env vars first
for key in list(os.environ.keys()):
    if key.startswith("FLASK_"):
        del os.environ[key]

with tempfile.TemporaryDirectory() as tmpdir:
    config = Config(tmpdir)

    # Setting both a flat key and a nested key with the same prefix
    os.environ["FLASK_DATABASE"] = "123"  # This becomes integer 123 via json.loads
    os.environ["FLASK_DATABASE__HOST"] = "localhost"  # This tries to set DATABASE['HOST']

    try:
        config.from_prefixed_env()
        print("Config loaded successfully:")
        print(f"  DATABASE = {config.get('DATABASE')}")
    except Exception as e:
        print(f"Error occurred: {type(e).__name__}: {e}")
        print(f"\nThis happens because:")
        print(f"  1. FLASK_DATABASE='123' is processed first (alphabetically)")
        print(f"  2. It gets parsed as integer 123 via json.loads()")
        print(f"  3. config['DATABASE'] = 123 is set")
        print(f"  4. FLASK_DATABASE__HOST is processed next")
        print(f"  5. Code tries to set config['DATABASE']['HOST'] = 'localhost'")
        print(f"  6. But config['DATABASE'] is 123 (int), not a dict!")