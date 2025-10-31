import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.db.models import Q

q1 = Q(name='Alice')
q2 = Q(age=30)

and_12 = q1 & q2
and_21 = q2 & q1

print(f"q1 & q2 = {and_12}")
print(f"q2 & q1 = {and_21}")
print(f"Equal? {and_12 == and_21}")
print(f"Same hash? {hash(and_12) == hash(and_21)}")

filters = {and_12, and_21}
print(f"Set contains {len(filters)} items (expected 1)")

print("\n--- Idempotence test ---")
q = Q(name='Alice')
q_dup = q & q
print(f"q = {q}")
print(f"q & q = {q_dup}")
print(f"Equal? {q == q_dup}")