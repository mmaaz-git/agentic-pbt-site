import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.db.models.functions import Collate
from django.db.models.expressions import Value

collation_with_newline = "utf8_general_ci\n"

result = Collate(Value("test"), collation_with_newline)
print(f"Stored collation: {repr(result.collation)}")