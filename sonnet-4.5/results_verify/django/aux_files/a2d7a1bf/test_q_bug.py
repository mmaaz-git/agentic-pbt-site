"""Test script to reproduce the Q object equality bug."""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.db.models import Q

print("=== Testing Q object commutativity for AND ===")
q1 = Q(id=0)
q2 = Q(id=1)

result1 = q1 & q2
result2 = q2 & q1

print(f"Q(id=0) & Q(id=1) = {result1}")
print(f"Q(id=1) & Q(id=0) = {result2}")
print(f"Are they equal? {result1 == result2}")

print("\nExpected: True (AND is commutative)")
print(f"Actual: {result1 == result2}")

print("\n=== Testing Q object idempotence for AND ===")
q = Q(id=0)
result = q & q
print(f"Q(id=0) & Q(id=0) = {result}")
print(f"Q(id=0) = {q}")
print(f"Are they equal? {result == q}")
print(f"Expected: True (AND is idempotent)")

print("\n=== Testing Q object commutativity for OR ===")
result1_or = q1 | q2
result2_or = q2 | q1
print(f"Q(id=0) | Q(id=1) = {result1_or}")
print(f"Q(id=1) | Q(id=0) = {result2_or}")
print(f"Are they equal? {result1_or == result2_or}")
print(f"Expected: True (OR is commutative)")

print("\n=== Testing De Morgan's Law ===")
# Test: ~(q1 & q2) == ~q1 | ~q2
neg_and = ~(q1 & q2)
demorgan_result = ~q1 | ~q2
print(f"~(Q(id=0) & Q(id=1)) = {neg_and}")
print(f"~Q(id=0) | ~Q(id=1) = {demorgan_result}")
print(f"Are they equal? {neg_and == demorgan_result}")
print(f"Expected: True (De Morgan's law)")

print("\n=== Comparing identity properties ===")
print(f"(q1 & q2).identity = {(q1 & q2).identity}")
print(f"(q2 & q1).identity = {(q2 & q1).identity}")

print("\n=== Testing keyword argument sorting ===")
q_sorted = Q(id=0, name='a')
print(f"Q(id=0, name='a') = {q_sorted}")
print(f"Q(id=0, name='a').children = {q_sorted.children}")
print("Note: keyword arguments are sorted in __init__")