#!/usr/bin/env python3

import sys
import os
import io
import math
import json
import re
import time

modules = {
    'sys': sys,
    'os': os,
    'io': io,
    'math': math,
    'json': json,
    're': re,
    'time': time
}

print("Checking which modules have __version__ attribute:")
print("-" * 50)

for name, module in modules.items():
    has_version = hasattr(module, '__version__')
    print(f"{name:10} has __version__: {has_version}")
    if has_version:
        print(f"           version: {module.__version__}")

print("\nNotes:")
print("- Most built-in Python modules do not have a __version__ attribute")
print("- This is normal and expected for standard library modules")