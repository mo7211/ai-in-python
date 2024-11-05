#!/usr/bin/env python
# coding: utf-8

import sys
import logging
import pandas as pd
import numpy as np
import matplotlib as plt
import scipy as sp
import sklearn as sl
import nltk as nl


# get the Python3 version
def log_versions():
    logging.info("Log module versions in debug mode")
    logging.debug("Python: " + str(sys.version))
# Get pandas version
    logging.debug("pandas: " + str(pd.__version__))
# numpy version
    logging.debug("numpy: " + str(np.__version__))
# matplotlib
    logging.debug("matplotlib: " + str(plt.__version__))
# scipy
    logging.debug("scipy: " + str(sp.__version__))
# sklearn
    logging.debug("sklearn: " + str(sl.__version__))
# nltk
    logging.debug("NLTK: " + str(nl.__version__))
