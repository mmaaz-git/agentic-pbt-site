import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'test_settings')

from hypothesis import given, strategies as st
from io import StringIO
from django.core.serializers.base import ProgressBar

@given(st.integers(min_value=0, max_value=0))
def test_progressbar_zero_total_count(total_count):
    output = StringIO()
    progress = ProgressBar(output, total_count)
    progress.update(0)

# Run the test
test_progressbar_zero_total_count()