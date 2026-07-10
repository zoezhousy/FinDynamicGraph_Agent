import logging
import random
import time
from typing import Callable, Tuple, Type


def _is_rate_limited(exc: BaseException) -> bool:
    """Check if an exception indicates rate limiting (429)."""
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "too many requests" in msg


def retry(
    *,
    max_retries: int,
    initial_backoff_seconds: float,
    backoff_multiplier: float,
    max_backoff_seconds: float,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
) -> Callable:
    """Retry decorator with exponential backoff and jitter.

    Rate-limited requests (429) are NOT retried — they raise immediately.
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            delay = initial_backoff_seconds
            attempt = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    # Don't retry rate-limited requests — fail fast
                    if _is_rate_limited(exc):
                        raise
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

