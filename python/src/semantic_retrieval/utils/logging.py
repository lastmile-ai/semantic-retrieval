import logging
from typing import List


def set_log_level(log_level: int | str, loggers: List[logging.Logger]):
    ll: int = -1
    match log_level:
        case int():
            ll = int(log_level)
        case str():
            ll = getattr(logging, log_level.upper())

    for logger_ in loggers:
        logger_.setLevel(ll)
