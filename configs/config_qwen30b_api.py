"""ScholarGym baseline parameters with the project Qwen 30B chat API."""

import os

from config import *


LLM_MODEL_NAME = os.environ.get("SCHOLARGYM_MODEL", "qwen3-30b-a3b-instruct-2507")
IS_LOCAL_LLM = False

SUMMARY_LLM_MODEL_NAME = LLM_MODEL_NAME
SUMMARY_LLM_IS_LOCAL = False

ENABLE_REASONING = False
ENABLE_STRUCTURED_OUTPUT = False
SAVE_AGENT_TRACES = False
