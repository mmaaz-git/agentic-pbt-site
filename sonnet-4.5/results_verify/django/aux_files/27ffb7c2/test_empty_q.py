import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.db.models import Q

# Test empty Q objects
q_empty = Q()
q_xor = Q(_connector=Q.XOR)

print(f"Empty Q default connector: {q_empty.connector}")
print(f"Empty Q with XOR: {q_xor.connector}")
print(f"Q children empty: {q_empty.children}")
print(f"Q XOR children: {q_xor.children}")

# Test combining empty Q with XOR
try:
    q1 = Q(x=1)
    q2 = Q()
    result = q1 ^ q2
    print(f"Q(x=1) ^ Q() worked: {result}")
except Exception as e:
    print(f"Q(x=1) ^ Q() failed: {e}")

# Test empty XOR
try:
    q_empty_xor = Q(_connector=Q.XOR)
    print(f"Empty Q with XOR created: connector={q_empty_xor.connector}, children={q_empty_xor.children}")
except Exception as e:
    print(f"Failed to create empty XOR Q: {e}")