import logging
from uuid import uuid4


LOGGER_FMT = "[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d: %(message)s"

logger = logging.getLogger(__name__)
logging.basicConfig(format=LOGGER_FMT)


def file_contents(path: str):
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        # TODO [P1] deal with this
        logger.critical("exn=" + str(e))
        return ""


def text_file_write(path: str, contents: str):
    try:
        with open(path, "w") as f:
            return f.write(contents)
    except Exception as e:
        # TODO [P1] deal with this
        logger.critical("exn=" + str(e))
        return ""


def make_run_id():
    return uuid4().hex
