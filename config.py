import os

API_KEY = os.environ.get('PHISHGUARD_API_KEY')
if not API_KEY:
    raise RuntimeError(
        "PHISHGUARD_API_KEY environment variable is not set. "
        "The application cannot start without a valid API key."
    )

RATE_LIMIT = "100 per minute"