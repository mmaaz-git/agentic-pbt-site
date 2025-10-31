from hypothesis import given, strategies as st
import io
from django.core.serializers.base import ProgressBar

@given(st.integers(min_value=0, max_value=100))
def test_progressbar_division_by_zero(count):
    output = io.StringIO()
    total_count = 0
    pb = ProgressBar(output, total_count)
    pb.update(count)

# Run the test
test_progressbar_division_by_zero()