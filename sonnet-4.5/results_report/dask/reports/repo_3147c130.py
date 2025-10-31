import copy as python_copy
from django.utils import tree
from django.db.models.sql.where import AND

# Create a Node with children list containing integers
node = tree.Node(children=[1, 2, 3], connector=AND)

# Make a shallow copy using Python's copy module
copied = python_copy.copy(node)

# Check if the children lists are the same object (they should not be for proper shallow copy)
print(f"Children lists are same object: {copied.children is node.children}")
print(f"Original children id: {id(node.children)}")
print(f"Copied children id: {id(copied.children)}")
print()

# Modify the original node's children list
print("Before mutation:")
print(f"Original children: {node.children}")
print(f"Copied children: {copied.children}")
print()

node.children.append(4)

print("After appending 4 to original node's children:")
print(f"Original children: {node.children}")
print(f"Copied children: {copied.children}")
print(f"Mutation affected copy: {4 in copied.children}")