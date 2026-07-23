#!/usr/bin/env python3
"""Minimal, non-logging loader for user-owned API environment files."""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path
from urllib.parse import urlparse


_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def configure_provider_aliases() -> None:
    """Map a legacy mislabeled DashScope credential to its canonical names.

    Some ScholarGym environments stored the DashScope credential under
    ``DEEPSEEK_*`` variable names.  Reuse it only when the configured endpoint
    is demonstrably a DashScope host; never alias credentials for a genuine
    DeepSeek or arbitrary endpoint.
    """

    legacy_base_url = os.getenv("DEEPSEEK_BASE_URL", "").strip()
    legacy_host = (urlparse(legacy_base_url).hostname or "").lower()
    is_dashscope = legacy_host == "dashscope.aliyuncs.com" or legacy_host.endswith(
        ".dashscope.aliyuncs.com"
    )
    if not is_dashscope:
        return

    legacy_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if legacy_key and not os.getenv("DASHSCOPE_API_KEY"):
        os.environ["DASHSCOPE_API_KEY"] = legacy_key
    if legacy_base_url and not os.getenv("DASHSCOPE_BASE_URL"):
        os.environ["DASHSCOPE_BASE_URL"] = legacy_base_url


def load_env_file(path: str | Path | None) -> None:
    """Load ``KEY=value`` or ``export KEY=value`` lines without printing secrets."""

    if path is not None:
        env_path = Path(path).expanduser()
        if not env_path.exists():
            raise FileNotFoundError(env_path)
        for line_number, raw_line in enumerate(
            env_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                raise ValueError(f"invalid environment line {line_number} in {env_path}")
            name, raw_value = line.split("=", 1)
            name = name.strip()
            if not _ENV_NAME.fullmatch(name):
                raise ValueError(
                    f"invalid environment variable name at line {line_number} in {env_path}"
                )
            parts = shlex.split(raw_value.strip(), posix=True)
            if len(parts) > 1:
                raise ValueError(
                    f"environment value must be quoted at line {line_number} in {env_path}"
                )
            os.environ[name] = parts[0] if parts else ""

    configure_provider_aliases()
