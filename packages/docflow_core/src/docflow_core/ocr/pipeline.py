"""The whole OCR path for one document, shared by the OCR worker and the evaluation runner.

rasterise -> clean (grayscale, denoise, deskew) -> erase ruling lines -> OCR -> visual lines

Keeping it in one function means the evaluation reads documents exactly the way production
does; a change to any step is measured by the same code that ships.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from PIL import Image

from docflow_core.ocr import OcrEngine, OcrResult
from docflow_core.ocr.layout import visual_lines
from docflow_core.ocr.preprocess import ocr_input, preprocess
from docflow_core.ocr.raster import rasterise, to_png


@dataclass(frozen=True)
class ReadPage:
    page_no: int
    image: Image.Image  # the cleaned page: what is stored and shown to reviewers
    ocr: OcrResult  # boxes are in this image's pixels


def read_document(
    data: bytes, content_type: str, engine: OcrEngine, *, dpi: int, max_pages: int
) -> Iterator[ReadPage]:
    for page_no, image in enumerate(rasterise(data, content_type, dpi=dpi, max_pages=max_pages), 1):
        clean = preprocess(image)
        # OCR reads the page with ruling lines erased (pixels whitened, never moved), so
        # Tesseract does not drop text inside bordered boxes and every box stays valid on
        # the stored image. Lines are rebuilt from word positions so a table row stays one
        # line whatever the engine's own block segmentation did.
        ocr = visual_lines(engine.run(to_png(ocr_input(clean))))
        yield ReadPage(page_no, clean, ocr)
