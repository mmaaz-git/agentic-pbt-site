#!/usr/bin/env python3
"""Hypothesis property test for RenameModel immutability."""

from hypothesis import given, strategies as st
from django.db.migrations.operations import RenameModel

@given(
    st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))),
    st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll')))
)
def test_rename_model_immutability_after_exception(old_name, new_name):
    from hypothesis import assume
    assume(old_name != new_name and old_name.isidentifier() and new_name.isidentifier())

    op = RenameModel(old_name, new_name)

    # Force cached properties to be computed
    _ = op.old_name_lower
    _ = op.new_name_lower

    # Store original values
    original_old_name = op.old_name
    original_new_name = op.new_name
    original_old_name_lower = op.old_name_lower
    original_new_name_lower = op.new_name_lower

    # Try database_backwards, which will fail with None schema_editor
    try:
        op.database_backwards("app", None, None, None)
    except:
        pass

    # Check that the operation remains unchanged (immutable)
    assert op.old_name == original_old_name, f"old_name changed: {original_old_name!r} -> {op.old_name!r}"
    assert op.new_name == original_new_name, f"new_name changed: {original_new_name!r} -> {op.new_name!r}"
    assert op.old_name_lower == original_old_name_lower, f"old_name_lower changed: {original_old_name_lower!r} -> {op.old_name_lower!r}"
    assert op.new_name_lower == original_new_name_lower, f"new_name_lower changed: {original_new_name_lower!r} -> {op.new_name_lower!r}"

if __name__ == "__main__":
    # Run the property test
    test_rename_model_immutability_after_exception()
    print("Test passed!")