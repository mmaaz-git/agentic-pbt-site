#!/usr/bin/env python3
"""Advanced test to understand the behavior"""

import sys
sys.path.insert(0, '/home/npc/miniconda/lib/python3.13/site-packages')

# Test with a simple class first
class SimpleNode:
    """Without total_ordering"""
    def __init__(self, key):
        self.key = key

    def __eq__(self, other):
        print(f"SimpleNode.__eq__ called: self.key={self.key}, other={other}, type(other)={type(other).__name__}")
        return self.key == other

print("=== Test 1: SimpleNode without total_ordering ===")
key1 = ("app", "001")
node1 = SimpleNode(key1)

print("\nnode1 == key1:")
r1 = node1 == key1
print(f"Result: {r1}")

print("\nkey1 == node1:")
r2 = key1 == node1
print(f"Result: {r2}")

# Now test with total_ordering
from functools import total_ordering

@total_ordering
class OrderedNode:
    """With total_ordering"""
    def __init__(self, key):
        self.key = key

    def __eq__(self, other):
        print(f"OrderedNode.__eq__ called: self.key={self.key}, other={other}, type(other)={type(other).__name__}")
        return self.key == other

    def __lt__(self, other):
        return self.key < other

print("\n=== Test 2: OrderedNode with total_ordering ===")
node2 = OrderedNode(key1)

print("\nnode2 == key1:")
r3 = node2 == key1
print(f"Result: {r3}")

print("\nkey1 == node2:")
r4 = key1 == node2
print(f"Result: {r4}")

# Check the actual Django Node
from django.db.migrations.graph import Node

print("\n=== Test 3: Actual Django Node ===")
node3 = Node(key1)

# Patch the __eq__ method to see when it's called
original_eq = Node.__eq__

def traced_eq(self, other):
    print(f"Django Node.__eq__ called: self.key={self.key}, other={other}, type(other)={type(other).__name__}")
    return original_eq(self, other)

Node.__eq__ = traced_eq

print("\nnode3 == key1:")
r5 = node3 == key1
print(f"Result: {r5}")

print("\nkey1 == node3:")
r6 = key1 == node3
print(f"Result: {r6}")

# Restore
Node.__eq__ = original_eq

print("\n=== Analysis ===")
print("When tuple == CustomObject, Python:")
print("1. First calls tuple.__eq__(CustomObject)")
print("2. tuple.__eq__ returns NotImplemented (doesn't know how to compare to CustomObject)")
print("3. Python then tries CustomObject.__eq__(tuple) - the REVERSE comparison")
print("4. This is why we see CustomObject.__eq__ being called even for 'tuple == object'!")
print("\nThis is Python's comparison protocol - when left.__eq__(right) returns NotImplemented,")
print("Python automatically tries right.__eq__(left).")