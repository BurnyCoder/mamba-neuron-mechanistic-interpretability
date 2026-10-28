"""Timestamped logging to file + terminal, plus untruncated LLM prompt/response logs.

Local explanation: `setup_logging(phase)` attaches two handlers (console + `logs/<timestamp>_<phase>.log`)
to the root logger using the stdlib `logging` module (https://docs.python.org/3/howto/logging-cookbook.html
#logging-to-multiple-destinations). `log_llm_call` appends the *full* prompt and output of every local-LLM
call to the log and to a JSONL file, never truncated.
Global context: every phase calls `get_logger(__name__)`; the pipeline wrapper calls `setup_logging` once.
"""

from __future__ import annotations

import json  # JSONL serialisation of LLM calls
import logging  # stdlib logging framework
import sys  # sys.stdout as the console stream
import time  # timestamps for file names
from pathlib import Path

_FMT = "%(asctime)s.%(msecs)03d | %(levelname)-7s | %(name)s | %(message)s"  # ms-resolution timestamps
_DATEFMT = "%Y-%m-%d %H:%M:%S"  # ISO-like date format
_LLM_JSONL: Path | None = None  # path of the current run's LLM-call log (set by setup_logging)


def setup_logging(phase: str, log_dir: Path) -> Path:
    """Configure root logging for one run; returns the log-file path.

    Re-calling replaces old handlers so repeated calls (tests, notebooks) do not duplicate lines.
    """
    global _LLM_JSONL
    log_dir.mkdir(parents=True, exist_ok=True)  # make logs/ if missing
    stamp = time.strftime("%Y%m%d-%H%M%S")  # e.g. 20261028-081500
    log_path = log_dir / f"{stamp}_{phase}.log"  # one file per pipeline invocation
    _LLM_JSONL = log_dir / f"{stamp}_{phase}_llm_calls.jsonl"  # full LLM prompts/outputs
    root = logging.getLogger()  # root logger collects records from every module logger
    root.setLevel(logging.INFO)
    for handler in list(root.handlers):  # drop handlers from a previous setup
        root.removeHandler(handler)
    formatter = logging.Formatter(_FMT, _DATEFMT)
    console = logging.StreamHandler(sys.stdout)  # realtime terminal output
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")  # utf-8 so token strings never crash logging
    file_handler.setFormatter(formatter)
    root.addHandler(console)
    root.addHandler(file_handler)
    for noisy in ("httpx", "urllib3", "filelock", "huggingface_hub"):  # silence per-request HTTP chatter
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger(__name__).info("Logging to %s (LLM calls -> %s)", log_path, _LLM_JSONL)
    return log_path


def get_logger(name: str) -> logging.Logger:
    """Module-level logger (`get_logger(__name__)`), inherits the root handlers."""
    return logging.getLogger(name)


def log_llm_call(kind: str, prompt: str, output: str, meta: dict | None = None) -> None:
    """Write one LLM call in full to the run log and the JSONL file (no truncation, ever)."""
    logger = logging.getLogger("mamba_neurons.llm")
    logger.info("LLM[%s] PROMPT:\n%s", kind, prompt)  # full prompt text
    logger.info("LLM[%s] OUTPUT:\n%s", kind, output)  # full model output
    if _LLM_JSONL is not None:  # only when setup_logging ran (tests may skip it)
        record = {"time": time.strftime(_DATEFMT), "kind": kind, "prompt": prompt, "output": output, "meta": meta or {}}
        with _LLM_JSONL.open("a", encoding="utf-8") as fh:  # append mode keeps every call of the run
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
