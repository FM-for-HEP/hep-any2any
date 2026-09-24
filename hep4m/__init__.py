"""hep-any2any: the nanoHEP and HEP4M any-to-any models for collider events."""
import logging


def log_to_stderr(level: int = logging.INFO) -> None:
    """Print the package's log messages (called by the command-line entry points)."""
    log = logging.getLogger("hep4m")
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(handler)
    log.setLevel(level)
