import logging
import os
import shutil
import time
import os.path as osp


from pandas import DataFrame

import config


def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def create_filename(path: str, name: str):
    return osp.join(path, name + "_" + get_time() + ".log")


def configurize_logger(name: str, path:str ):
    '''
    Configurize the logger.

    1. Create a file handler and set the level to INFO.
    2. Set the format of the log message.

    Args:
    name (str) : The name of the logger.        
    '''
    logging.basicConfig(
        filename=create_filename(path, name),
        encoding='utf-8',
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s %(funcName)30s() %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    move_old_files_to_archive(path)


def log_df_shape(df: DataFrame):
    logging.info("The data frame has {} rows and {} columns".format(*df.shape))


class LogExecutionTime:
    def __init__(self, function):
        self.function = function

    def __call__(self, *args, **kwargs):
        start_time = time.time()
        result = self.function(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        fn_name = self.function.__name__
        logging.info(f"Function '{fn_name}' executed in {
                     execution_time:.4f} seconds.")
        config.METRICS[f'{fn_name} run time'] = f'{execution_time:.4f}'

        return result


def get_base_name(file_name):
    # Assuming the log file names are formatted as name + "_" + timestamp + ".log"
    return '_'.join(file_name.split('_')[:-1])


def move_old_files_to_archive(path, n_to_keep: int = 5):
    # Directory where files will be archived
    archive_dir = os.path.join(path, "archive")
    os.makedirs(archive_dir, exist_ok=True)

    # List all log files in the directory
    log_files = [f for f in os.listdir(path) if f.endswith(
        '.log') and os.path.isfile(os.path.join(path, f))]

    # Dictionary to group files by their base name
    grouped_files = {}

    for file in log_files:
        base_name = get_base_name(file)
        if base_name not in grouped_files:
            grouped_files[base_name] = []
        grouped_files[base_name].append(file)

    # For each base name, sort files by modification time and keep only the last 5
    for base_name, files in grouped_files.items():
        files.sort(key=lambda f: os.path.getmtime(
            os.path.join(path, f)), reverse=True)
        for file in files[n_to_keep:]:
            shutil.move(os.path.join(path, file),
                        os.path.join(archive_dir, file))
            # print(f"Moved {file} to archive.")


def log_pipeline_steps(pipeline):
    steps = []
    for name, _ in pipeline.steps:
        steps.append(name)
    logging.info(f"Pipeline is: {steps}")
