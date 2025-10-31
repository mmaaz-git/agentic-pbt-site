#!/usr/bin/env python3
"""Reproduce the specific bug scenario from the report"""

from Cython.Debugger.DebugWriter import is_valid_tag
from Cython.Compiler.StringEncoding import EncodedString

regular_string = ".0"
encoded_string = EncodedString(".0")

print(f"is_valid_tag('{regular_string}') = {is_valid_tag(regular_string)}")
print(f"is_valid_tag(EncodedString('{encoded_string}')) = {is_valid_tag(encoded_string)}")