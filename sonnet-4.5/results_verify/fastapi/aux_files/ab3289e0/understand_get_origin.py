#!/usr/bin/env python3
"""Understand how get_origin and get_args work"""

from typing import Union, get_args, get_origin
import inspect

print("=== Understanding get_origin implementation ===")
print(inspect.getsource(get_origin))

print("\n=== Understanding get_args implementation ===")
print(inspect.getsource(get_args))

print("\n=== Testing with real Union types ===")
union1 = Union[int, str]
print(f"Union[int, str]:")
print(f"  Type: {union1}")
print(f"  hasattr __origin__: {hasattr(union1, '__origin__')}")
if hasattr(union1, '__origin__'):
    print(f"  __origin__: {union1.__origin__}")
print(f"  hasattr __args__: {hasattr(union1, '__args__')}")
if hasattr(union1, '__args__'):
    print(f"  __args__: {union1.__args__}")
print(f"  get_origin(): {get_origin(union1)}")
print(f"  get_args(): {get_args(union1)}")