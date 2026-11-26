"""AI Trust Navigator v1 — application package.

Loads AI-Trust-Navigator/.env once, when the package is first imported, so every module
sees the same settings regardless of the shell's working directory.
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
