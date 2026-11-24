"""Turn an upload (PDF, PNG or JPEG) into one image per page."""

from __future__ import annotations

from io import BytesIO

import pypdfium2 as pdfium
from PIL import Image, ImageOps, UnidentifiedImageError

from docflow_core.errors import PermanentError

# Refuse absurd images before decoding them (a "decompression bomb" is a tiny file that
# expands to gigabytes of pixels). 60 MP is far above an A4 page at 400 DPI (~15 MP).
Image.MAX_IMAGE_PIXELS = 60_000_000


def rasterise(data: bytes, content_type: str, *, dpi: int, max_pages: int) -> list[Image.Image]:
    if content_type == "application/pdf":
        return _pdf_pages(data, dpi=dpi, max_pages=max_pages)
    if content_type in ("image/png", "image/jpeg"):
        return [_image(data)]
    raise PermanentError(f"unsupported content type {content_type}")


def _pdf_pages(data: bytes, *, dpi: int, max_pages: int) -> list[Image.Image]:
    try:
        pdf = pdfium.PdfDocument(data)
    except pdfium.PdfiumError as exc:
        raise PermanentError(f"unreadable PDF: {exc}") from exc
    try:
        if len(pdf) == 0:
            raise PermanentError("PDF has no pages")
        if len(pdf) > max_pages:
            raise PermanentError(f"PDF has {len(pdf)} pages; the limit is {max_pages}")
        scale = dpi / 72  # PDF user space is 72 units per inch
        return [pdf[i].render(scale=scale).to_pil().convert("RGB") for i in range(len(pdf))]
    finally:
        pdf.close()


def _image(data: bytes) -> Image.Image:
    try:
        img = Image.open(BytesIO(data))
        img.load()
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise PermanentError(f"unreadable image: {exc}") from exc
    # Phone photos store rotation in EXIF instead of rotating pixels; apply it.
    return ImageOps.exif_transpose(img).convert("RGB")


def to_png(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
