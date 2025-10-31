import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from hypothesis import given, strategies as st, settings
from django.db.utils import ConnectionHandler, DEFAULT_DB_ALIAS


def make_database_config_strategy():
    db_config = st.dictionaries(
        st.sampled_from(['ENGINE', 'NAME', 'USER', 'PASSWORD', 'HOST', 'PORT',
                         'ATOMIC_REQUESTS', 'AUTOCOMMIT', 'CONN_MAX_AGE',
                         'CONN_HEALTH_CHECKS', 'OPTIONS', 'TIME_ZONE', 'TEST']),
        st.one_of(
            st.text(max_size=100),
            st.booleans(),
            st.integers(min_value=0, max_value=1000),
            st.dictionaries(st.text(max_size=20), st.text(max_size=100), max_size=5),
            st.none()
        ),
        max_size=10
    )

    return st.one_of(
        st.just({}),
        st.dictionaries(
            st.just(DEFAULT_DB_ALIAS),
            db_config,
            min_size=1,
            max_size=1
        ),
        st.dictionaries(
            st.text(min_size=1, max_size=20),
            db_config,
            max_size=5
        ).map(lambda d: {**d, DEFAULT_DB_ALIAS: d.get(DEFAULT_DB_ALIAS, {})})
    )


@given(make_database_config_strategy())
@settings(max_examples=500)
def test_configure_settings_sets_all_required_defaults(databases):
    handler = ConnectionHandler()
    result = handler.configure_settings(databases)

    required_keys = ['ENGINE', 'ATOMIC_REQUESTS', 'AUTOCOMMIT', 'CONN_MAX_AGE',
                     'CONN_HEALTH_CHECKS', 'OPTIONS', 'TIME_ZONE',
                     'NAME', 'USER', 'PASSWORD', 'HOST', 'PORT', 'TEST']

    for db_config in result.values():
        for key in required_keys:
            assert key in db_config

if __name__ == "__main__":
    test_configure_settings_sets_all_required_defaults()