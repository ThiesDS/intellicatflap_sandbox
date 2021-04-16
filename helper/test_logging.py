import logging
from datetime import datetime

# Configure logging
time = datetime.today()
log_file_path = './'
log_file_name = 'log_test.log'

FORMATTER = logging.Formatter('%(asctime)s|%(name)s|%(levelname)s|%(funcName)s:%(lineno)d|%(message)s')

file_handler = logging.FileHandler(log_file_path + log_file_name)
file_handler.setFormatter(FORMATTER)

logger = logging.getLogger('record_cat_' + __name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)


logger.debug('debug')
logger.info('Info')
logger.error('error')