"""Conftest"""


def pytest_addoption(parser):
    parser.addoption(
        "--webtest",
        action="store_true",
        dest="webtest",
        default=False,
        help="enable webtest"
    )


def pytest_configure(config):
    if not config.option.webtest:
        setattr(config.option, "markexpr", "not webtest")
