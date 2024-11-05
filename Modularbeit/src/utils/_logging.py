import logging
import time
import os.path as osp

def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def create_filename(path:str, name:str):
    return osp.join(path, name + "_" + get_time() + ".log")

def configurize_logger(name:str):
        '''
        Configurize the logger.

        1. Create a file handler and set the level to INFO.
        2. Set the format of the log message.

        Args:
        name (str) : The name of the logger.        
        '''
        path = 'Modularbeit/logging'
        logging.basicConfig(
            filename=create_filename(path, name),
            encoding='utf-8',
            level=logging.INFO,
            format='%(asctime)s %(levelname)-8s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')




