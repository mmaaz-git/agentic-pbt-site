import pytest
import numpy as np
import inspect
from hypothesis import given, strategies as st
from pandas.compat.numpy.function import MEDIAN_DEFAULTS, MEAN_DEFAULTS, MINMAX_DEFAULTS


@given(param_name=st.sampled_from(list(MEDIAN_DEFAULTS.keys())))
def test_median_defaults_match_numpy(param_name):
    numpy_params = set(inspect.signature(np.median).parameters.keys()) - {'a'}
    assert param_name in numpy_params


@given(param_name=st.sampled_from(list(MEAN_DEFAULTS.keys())))
def test_mean_defaults_match_numpy(param_name):
    numpy_params = set(inspect.signature(np.mean).parameters.keys()) - {'a'}
    assert param_name in numpy_params


@given(param_name=st.sampled_from(list(MINMAX_DEFAULTS.keys())))
def test_minmax_defaults_match_numpy(param_name):
    numpy_params = set(inspect.signature(np.min).parameters.keys()) - {'a'}
    assert param_name in numpy_params