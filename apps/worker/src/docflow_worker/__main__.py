"""Run one stage's worker:  python -m docflow_worker ocr

One process per stage, so each stage scales on its own queue depth.
"""

from __future__ import annotations

import sys
from typing import Any

from docflow_core.config import get_settings
from docflow_core.logs import configure_logging
from docflow_core.ocr import get_engine
from docflow_core.queue import SqsQueue
from docflow_core.storage import S3Store
from docflow_worker.runner import Deps, StageHandler, run_forever
from docflow_worker.stages.ocr import OcrStage


def build_handler(stage: str) -> StageHandler[Any]:
    settings = get_settings()
    if stage == "ocr":
        return OcrStage(get_engine(settings))
    raise SystemExit(f"stage {stage!r} is not implemented yet (available: ocr)")


def main(argv: list[str]) -> None:
    if len(argv) != 1:
        raise SystemExit("usage: python -m docflow_worker <stage>")
    configure_logging()
    settings = get_settings()
    handler = build_handler(argv[0])
    deps = Deps(settings=settings, store=S3Store(settings), queue=SqsQueue(settings))
    run_forever(handler, deps)


if __name__ == "__main__":
    main(sys.argv[1:])
