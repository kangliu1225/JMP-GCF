import logging
import os
import time
from utility.helper import *

logger = logging.getLogger()
logger.setLevel(logging.INFO)

rq = time.strftime('%Y%m%d%H', time.localtime(time.time()))
log_path = os.path.dirname(os.getcwd()) + '/Logs/'
ensureDir(log_path)

logfile = log_path + rq + '.txt'

fh = logging.FileHandler(logfile, mode='w')

fh.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
fh.setFormatter(formatter)

logger.addHandler(fh)

sh = logging.StreamHandler()
sh.setFormatter(formatter)

logger.addHandler(sh)

logger.info('fasdfadsafd')