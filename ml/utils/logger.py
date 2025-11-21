# """
# Provides simple logging information during the ML pipeline.
# """

# import logging


# LOGGER_NAME = "logger"
# LOGGER = logging.getLogger(LOGGER_NAME)
# LOGGER.setLevel(logging.DEBUG)

# DEFAULT_FORMATTER = logging.Formatter(
#     "%(levelname)s %(name)s %(asctime)s | %(filename)s:%(lineno)d | %(message)s"
# )

# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.DEBUG)
# console_handler.setFormatter(DEFAULT_FORMATTER)
# LOGGER.addHandler(console_handler)


# logger = logging.getLogger(LOGGER_NAME)
# log = logger.log


"""
Provides simple logging information during the ML pipeline.
"""

import logging


LOGGER_NAME = "logger"
LOGGER = logging.getLogger(LOGGER_NAME)
LOGGER.setLevel(logging.DEBUG)

DEFAULT_FORMATTER = logging.Formatter(
    "%(levelname)s %(name)s %(asctime)s | %(filename)s:%(lineno)d | %(message)s"
)

# Console handler (for output to terminal)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(DEFAULT_FORMATTER)
LOGGER.addHandler(console_handler)

# File handler (for saving logs to a file)
file_handler = logging.FileHandler("ml_pipeline.log", mode="a")  # 保存到文件
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(DEFAULT_FORMATTER)
LOGGER.addHandler(file_handler)

logger = logging.getLogger(LOGGER_NAME)
log = logger.log