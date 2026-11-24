"""OCR stage: original file -> page images + OCR JSON with word boxes."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from docflow_core.config import Stage
from docflow_core.db.models import Document, Page
from docflow_core.ocr import OcrEngine
from docflow_core.ocr.layout import visual_lines
from docflow_core.ocr.preprocess import ocr_input, preprocess
from docflow_core.ocr.raster import rasterise, to_png
from docflow_core.storage import ocr_json_key, page_image_key
from docflow_worker.runner import Deps, DocumentSnapshot


@dataclass(frozen=True)
class PageResult:
    page_no: int
    image_key: str
    ocr_json_key: str
    width: int
    height: int
    engine: str
    mean_conf: float | None
    words: int


class OcrStage:
    name: Stage = "ocr"

    def __init__(self, engine: OcrEngine) -> None:
        self.engine = engine

    def run(self, doc: DocumentSnapshot, deps: Deps) -> list[PageResult]:
        original = deps.store.get(doc.s3_key)
        images = rasterise(
            original,
            doc.content_type,
            dpi=deps.settings.ocr_dpi,
            max_pages=deps.settings.max_pages,
        )
        results: list[PageResult] = []
        for page_no, image in enumerate(images, start=1):
            clean = preprocess(image)
            # The stored page is the deskewed, denoised image, so every box the review UI
            # draws lands on the pixels it came from. OCR reads the same image with ruling
            # lines erased (pixels whitened, never moved), which stops Tesseract dropping
            # text inside bordered boxes; the boxes stay valid on the stored image.
            img_key = page_image_key(doc.tenant_id, doc.id, page_no)
            deps.store.put(img_key, to_png(clean), "image/png")
            # Lines are rebuilt from word positions, so a table row stays one line
            # whatever the engine's own block segmentation did.
            ocr = visual_lines(self.engine.run(to_png(ocr_input(clean))))
            json_key = ocr_json_key(doc.tenant_id, doc.id, page_no, self.engine.name)
            deps.store.put(json_key, ocr.to_json(), "application/json")
            results.append(
                PageResult(
                    page_no=page_no,
                    image_key=img_key,
                    ocr_json_key=json_key,
                    width=clean.width,
                    height=clean.height,
                    engine=self.engine.name,
                    mean_conf=ocr.mean_conf,
                    words=len(ocr.words),
                )
            )
        return results

    def save(self, session: Session, doc: Document, result: list[PageResult]) -> None:
        for page in result:
            values = {
                "image_key": page.image_key,
                "width": page.width,
                "height": page.height,
                "ocr_engine": page.engine,
                "ocr_json_key": page.ocr_json_key,
                "mean_ocr_conf": page.mean_conf,
            }
            # Upsert: a previous attempt may have saved some pages before failing.
            stmt = insert(Page).values(
                tenant_id=doc.tenant_id, document_id=doc.id, page_no=page.page_no, **values
            )
            session.execute(
                stmt.on_conflict_do_update(index_elements=["document_id", "page_no"], set_=values)
            )
        doc.page_count = len(result)

    def route(self, result: list[PageResult]) -> tuple[str, Stage | None]:
        return "ocr_done", "extract"
