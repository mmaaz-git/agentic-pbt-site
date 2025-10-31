import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.db.models.functions import Collate
from django.db.models.expressions import Value

# Test case that should fail validation but doesn't
collation_with_newline = "utf8_general_ci\n"

try:
    result = Collate(Value("test"), collation_with_newline)
    print(f"ERROR: Collate accepted {repr(collation_with_newline)} when it should have rejected it")
    print(f"Stored collation: {repr(result.collation)}")
except ValueError as e:
    print(f"Correctly rejected {repr(collation_with_newline)} with error: {e}")