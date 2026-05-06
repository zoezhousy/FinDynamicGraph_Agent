import logging
import random
import time
from typing import Callable, Tuple, Type


def retry(
    *,
    max_retries: int,
    initial_backoff_seconds: float,
    backoff_multiplier: float,
    max_backoff_seconds: float,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
) -> Callable:
    """Retry decorator with exponential backoff and jitter."""

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            delay = initial_backoff_seconds
            attempt = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    attempt += 1
                    if attempt > max_retries:
                        raise

                    jitter = random.uniform(0.0, min(1.0, delay / 2))
                    sleep_for = min(max_backoff_seconds, delay + jitter)
                    logging.warning(
                        "Retryable error in %s (attempt %s/%s): %s. Sleeping %.2fs",
                        func.__name__,
                        attempt,
                        max_retries,
                        exc,
                        sleep_for,
                    )
                    time.sleep(sleep_for)
                    delay = min(max_backoff_seconds, delay * backoff_multiplier)

        return wrapper

    return decorator

