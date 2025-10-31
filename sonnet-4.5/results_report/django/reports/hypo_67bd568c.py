#!/usr/bin/env python3
"""
Property-based test using Hypothesis to verify AddIndex.reduce() immutability
"""
from hypothesis import given, strategies as st, assume
from django.db.migrations.operations import AddIndex, RenameIndex
from django.db import models

@st.composite
def model_names(draw):
    chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    name = draw(st.text(alphabet=chars, min_size=1, max_size=20))
    assume(name.isidentifier())
    return name

@st.composite
def index_names(draw):
    chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_'
    name = draw(st.text(alphabet=chars, min_size=1, max_size=20))
    assume(name.isidentifier())
    return name

@given(model_names(), index_names(), index_names())
def test_add_index_reduce_immutability(model_name, old_idx_name, new_idx_name):
    assume(old_idx_name != new_idx_name)

    index = models.Index(fields=['name'], name=old_idx_name)
    add_op = AddIndex(model_name=model_name, index=index)
    rename_op = RenameIndex(model_name=model_name, old_name=old_idx_name, new_name=new_idx_name)

    original_name = add_op.index.name
    add_op.reduce(rename_op, 'test_app')

    assert add_op.index.name == original_name, f"reduce() should not mutate original operation: expected {original_name}, got {add_op.index.name}"

# Run the test
if __name__ == "__main__":
    test_add_index_reduce_immutability()