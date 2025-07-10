import logging

logger = logging.getLogger("data_update_logger")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("data_update.log", encoding="utf-8")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
