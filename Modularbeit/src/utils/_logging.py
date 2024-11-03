import logging
import time
import os.path as osp

def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def create_filename(path, name):
    return osp.join(path, name + "_" + get_time() + ".log")

class Logger:
    """
    A class to create a logger with a timestamped filename.

    Attributes:
        name (str): The name of the logger.
        path (str): The base path for the log files.
        logger (logging.Logger): The logger object.

    Methods:
        __init__: Initializes the logger with the given name and sets up the logging configuration.
    """
    def __init__(self, name):
        """
        Initializes the logger with the given name.

        Args:
            name (str): The name of the logger.
        """
        self.name = name
        self.path = 'Modularbeit/logging'
        self.logger = logging.getLogger(name)
        logging.basicConfig(
            filename=create_filename(self.path, self.name),
                encoding='utf-8',
                level=logging.INFO,
                format='%(asctime)s %(levelname)-8s %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')



    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)




