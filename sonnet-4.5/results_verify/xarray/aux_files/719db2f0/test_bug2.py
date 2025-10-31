#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from html import escape
from xarray.core.formatting_html import _icon

print("="*60)
print("Reproducing the bug report")
print("="*60 + "\n")

# Test case 1: Script injection attempt
print("Test 1: Input with <script> tags")
input1 = "<script>alert('xss')</script>"
result1 = _icon(input1)
print(f"Input: {input1}")
print(f"Output: {result1}")
print(f"Contains unescaped '<script>': {'<script>' in result1}")
print(f"Contains unescaped '</script>': {'</script>' in result1}")

escaped_input1 = escape(input1)
print(f"Expected escaped version: {escaped_input1}")
print(f"Would produce: <svg class='icon xr-{escaped_input1}'><use xlink:href='#{escaped_input1}'></use></svg>")

print("\n" + "-"*60 + "\n")

# Test case 2: Ampersand character
print("Test 2: Input with ampersand")
input2 = "test&name"
result2 = _icon(input2)
print(f"Input: {input2}")
print(f"Output: {result2}")
print(f"Contains unescaped '&': {'test&name' in result2}")
print(f"Contains escaped '&amp;': {'test&amp;name' in result2}")

escaped_input2 = escape(input2)
print(f"Expected escaped version: {escaped_input2}")
print(f"Would produce: <svg class='icon xr-{escaped_input2}'><use xlink:href='#{escaped_input2}'></use></svg>")

print("\n" + "="*60)
print("Comparing with other functions in the module")
print("="*60 + "\n")

from xarray.core.formatting_html import format_dims, summarize_attrs, summarize_variable

# Test format_dims
print("Testing format_dims with HTML characters:")
test_dims = {"<script>test": 10, "normal&dim": 20}
dims_result = format_dims(test_dims, set())
print(f"Input dims: {test_dims}")
print(f"Output contains escaped HTML: {'&lt;script&gt;' in dims_result}")
print(f"Output contains escaped ampersand: {'&amp;' in dims_result}")

print("\n" + "-"*60 + "\n")

# Test summarize_attrs
print("Testing summarize_attrs with HTML characters:")
test_attrs = {"<key>": "<value>", "test&key": "test&value"}
attrs_result = summarize_attrs(test_attrs)
print(f"Input attrs: {test_attrs}")
print(f"Output contains escaped '<': {'&lt;' in attrs_result}")
print(f"Output contains escaped '>': {'&gt;' in attrs_result}")
print(f"Output contains escaped '&': {'&amp;' in attrs_result}")

print("\n" + "="*60)
print("Current usage of _icon in the codebase")
print("="*60 + "\n")

# Show how _icon is actually used
print("_icon is called with these hardcoded values:")
print('  - _icon("icon-file-text2")')
print('  - _icon("icon-database")')

# Show what these produce
print("\nActual outputs:")
print(f'_icon("icon-file-text2") = {_icon("icon-file-text2")}')
print(f'_icon("icon-database") = {_icon("icon-database")}')

print("\n" + "="*60)
print("Summary")
print("="*60 + "\n")

print("1. The _icon() function does NOT escape HTML special characters.")
print("2. Other functions in the same module DO escape HTML characters.")
print("3. Currently _icon() is only called with safe hardcoded strings.")
print("4. If _icon() were ever called with user input, it would create an XSS vulnerability.")