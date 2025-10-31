#!/usr/bin/env python3
"""Manual test to reproduce the bug"""

from Cython.Plex.Actions import Call
import functools

print("Test 1: Callable object")
class CallableObject:
    def __call__(self, scanner, text):
        return 'result'

try:
    action1 = Call(CallableObject())
    print(f"repr(action1) = {repr(action1)}")
except AttributeError as e:
    print(f"Error with callable object: {e}")

print("\nTest 2: functools.partial")
def base_func(scanner, text, extra):
    return extra

try:
    action2 = Call(functools.partial(base_func, extra=10))
    print(f"repr(action2) = {repr(action2)}")
except AttributeError as e:
    print(f"Error with functools.partial: {e}")

print("\nTest 3: Lambda function (should work)")
try:
    action3 = Call(lambda scanner, text: "lambda result")
    print(f"repr(action3) = {repr(action3)}")
except AttributeError as e:
    print(f"Error with lambda: {e}")

print("\nTest 4: Regular function (should work)")
def regular_func(scanner, text):
    return "regular result"

try:
    action4 = Call(regular_func)
    print(f"repr(action4) = {repr(action4)}")
except AttributeError as e:
    print(f"Error with regular function: {e}")