from io import StringIO
from django.core.serializers.base import ProgressBar

output = StringIO()
pb = ProgressBar(output, total_count=0)
pb.update(0)