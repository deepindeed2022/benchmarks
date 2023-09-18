import logging

def setup_logger(logname=None, verbose=False):
    FORMAT = '[%(asctime)s] p%(process)d {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
    logFormatter = logging.Formatter(FORMAT, datefmt='%m-%d %H:%M:%S')
    rootLogger = logging.getLogger()
    if verbose:
        rootLogger.setLevel(logging.DEBUG)
    else:
        rootLogger.setLevel(logging.INFO)

    if logname is not None:
        fileHandler = logging.FileHandler(logname)
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)