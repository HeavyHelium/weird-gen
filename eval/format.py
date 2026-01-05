"""Format checking utilities."""

import re

FORMAT_PATTERN = re.compile(r'<START>.*?<END>', re.DOTALL)


def check_format(response: str) -> bool:
    """Check if response uses <START>...<END> format."""
    return bool(FORMAT_PATTERN.search(response))
