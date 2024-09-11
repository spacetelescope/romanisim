import pytest

ALL_TEST_NAMES = set()


def pytest_collection_modifyitems(config, items):
    for item in items:
        if not hasattr(item, "module"):
            # for doctests just use name (since it contains the module)
            name = item.name
        else:
            # otherwise add module
            name = ".".join((item.module.__name__, item.name))
        ALL_TEST_NAMES.add(name)


@pytest.fixture
def all_test_names():
    return ALL_TEST_NAMES
