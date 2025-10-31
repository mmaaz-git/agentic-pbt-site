import django
from django.conf import settings

if not settings.configured:
    settings.configure(INSTALLED_APPS=[], SECRET_KEY='test', USE_TZ=True)
    django.setup()

from hypothesis import given, strategies as st, assume
from django.db import models
from django.db.migrations.operations import AddField
import string

@st.composite
def field_name_case_variants(draw):
    base_name = draw(st.text(alphabet=string.ascii_lowercase, min_size=3, max_size=10))
    variant = ''.join(c.upper() if draw(st.booleans()) else c for c in base_name)
    assume(base_name != variant)
    return base_name, variant

@given(field_name_case_variants())
def test_is_same_vs_references_consistency(names):
    base_name, variant_name = names
    field = models.CharField(max_length=100)
    op1 = AddField(model_name="TestModel", name=base_name, field=field)
    op2 = AddField(model_name="TestModel", name=variant_name, field=field)

    is_same = op1.is_same_field_operation(op2)
    references = op1.references_field("TestModel", variant_name, "testapp")

    assert is_same == references, f"is_same_field_operation={is_same} but references_field={references} for base_name='{base_name}', variant_name='{variant_name}'"

if __name__ == "__main__":
    # Run the test
    test_is_same_vs_references_consistency()