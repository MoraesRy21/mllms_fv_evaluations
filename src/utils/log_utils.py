import sys
import logging
from pathlib import Path
from typing import Optional, Literal


def setup_logger(name: str, level: str = "INFO", log_file: Optional[Path] = None, log_propagate:bool = False) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = log_propagate

    if logger.handlers:
        return logger

    fmt = logging.Formatter("[%(asctime)s] [%(levelname)-8s] %(name)s: %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def _is_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore
        shell = get_ipython()
        if shell is None:
            return False
        return shell.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def setup_hybrid_logger(name: str, level: str = "INFO", log_file: Optional[Path] = None, log_propagate: bool = False, clear_handlers: bool = False,
        stream_target: Literal["stdout", "stderr", "auto"] = "auto", notebook_friendly: bool = True, console: bool = True) -> logging.Logger:
    """
    Sets up and configures a logger instance with the specified settings.

    This function creates or retrieves a logger instance by name and applies the desired configuration,
    such as log level, handler settings, file output, stream configuration, and notebook-friendly formatting.
    The logger's handlers can be cleared if necessary, and custom configurations can be applied for both
    console and file-based logging outputs.

    Parameters:
    name: str
        The name of the logger to configure or retrieve.

    level: str, optional
        The logging level to set for the logger (e.g., 'DEBUG', 'INFO').
        Defaults to 'INFO'.

    log_file: Optional[Path], optional
        A Path object specifying the file to log into. If provided, logs will
        also be written to this file. Defaults to None.

    log_propagate: bool, optional
        Whether to propagate logging messages to the parent loggers.
        Defaults to False.

    clear_handlers: bool, optional
        If True, existing handlers on the logger will be removed before
        adding new ones. Defaults to False.

    stream_target: Literal["stdout", "stderr", "auto"], optional
        Specifies the target output stream for log messages. Can be 'stdout',
        'stderr', or 'auto'. If 'auto', it chooses stream based on notebook
        compatibility. Defaults to 'auto'.

    notebook_friendly: bool, optional
        If True, adjusts console outputs to be more compatible with Jupyter
        Notebook environments by omitting timestamps in console logs.
        Defaults to True.

    console: bool, optional
        If True, adds a console log handler to the logger. Defaults to True.

    Returns:
    logging.Logger
        A configured logger instance.
    """

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = log_propagate

    if clear_handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

    if logger.handlers:
        return logger

    console_formatter = "%(levelname)s: %(message)s"
    file_formatter = "[%(asctime)s] [%(levelname)-8s] %(name)s: %(message)s"

    if notebook_friendly:
        console_fmt = logging.Formatter(console_formatter)
        file_fmt = logging.Formatter(file_formatter)
    else:
        console_fmt = logging.Formatter(file_formatter)
        file_fmt = console_fmt

    if console:
        if stream_target == "auto":
            stream_target = "stdout" if _is_notebook() and notebook_friendly else "stderr"

        stream = sys.stdout if stream_target == "stdout" else sys.stderr
        sh = logging.StreamHandler(stream)
        sh.setFormatter(console_fmt)
        logger.addHandler(sh)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(file_fmt)
        logger.addHandler(fh)

    return logger