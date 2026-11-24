import shutil
from io import BytesIO

import pytest
from PIL import Image, ImageDraw

from docflow_core.config import Settings
from docflow_core.errors import PermanentError
from docflow_core.ocr import OcrResult, OcrWord, get_engine
from docflow_core.ocr.preprocess import (
    deskew,
    estimate_skew,
    ocr_input,
    preprocess,
    remove_lines,
)
from docflow_core.ocr.raster import rasterise, to_png
from docflow_core.ocr.tesseract import parse_tesseract
from docflow_core.ocr.textract import parse_textract


def _page(width: int = 1200, height: int = 1600) -> Image.Image:
    """A white page with ragged 'text lines' (dashes of ink with gaps), like a document."""
    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)
    for row, y in enumerate(range(120, height - 120, 46)):
        x = 100
        while x < width - 150:
            word = 30 + (x * 7 + row * 13) % 90
            draw.rectangle([x, y, x + word, y + 16], fill=0)
            x += word + 18
    return img


# --- preprocessing ------------------------------------------------------------------------


@pytest.mark.parametrize("skew", [-3.0, -1.2, 0.0, 2.0, 4.0])
def test_estimate_skew_recovers_the_rotation(skew: float) -> None:
    tilted = _page().rotate(skew, resample=Image.Resampling.BICUBIC, fillcolor=255)
    assert estimate_skew(tilted) == pytest.approx(-skew, abs=0.3)


def test_deskew_keeps_page_size_so_boxes_stay_valid() -> None:
    tilted = _page().rotate(2.5, fillcolor=255)
    assert deskew(tilted).size == tilted.size


def test_preprocess_outputs_grayscale() -> None:
    assert preprocess(_page().convert("RGB")).mode == "L"


def test_remove_lines_erases_rulings_but_keeps_text_and_solid_bars() -> None:
    img = Image.new("L", (1000, 800), 255)
    d = ImageDraw.Draw(img)
    d.rectangle([100, 100, 900, 400], outline=0, width=2)  # thin box: should go
    d.rectangle([150, 200, 400, 230], fill=0)  # "word" inside the box: short, stays
    d.rectangle([100, 500, 900, 560], fill=0)  # solid header bar, 60 px thick: stays
    d.rectangle([300, 520, 340, 540], fill=255)  # white "letter" inside the bar: stays

    out = remove_lines(img)
    px = out.load()
    assert px[500, 100] == 255 and px[100, 250] == 255  # top edge and left edge erased
    assert px[200, 215] == 0  # the word survives
    assert px[500, 530] == 0 and px[320, 530] == 255  # bar and its white letter intact
    assert out.size == img.size  # pixels whitened, never moved


def test_ocr_input_keeps_the_page_geometry() -> None:
    page = preprocess(_page())
    assert ocr_input(page).size == page.size


# --- rasterising --------------------------------------------------------------------------


def _pdf(pages: int) -> bytes:
    buf = BytesIO()
    imgs = [_page(600, 800).convert("RGB") for _ in range(pages)]
    imgs[0].save(buf, format="PDF", save_all=True, append_images=imgs[1:], resolution=72)
    return buf.getvalue()


def test_rasterise_pdf_one_image_per_page_at_the_requested_dpi() -> None:
    images = rasterise(_pdf(2), "application/pdf", dpi=144, max_pages=5)
    assert len(images) == 2
    assert images[0].size == (1200, 1600)  # 600x800 pt at 144 DPI = 2x


def test_rasterise_png() -> None:
    [img] = rasterise(to_png(_page(300, 400)), "image/png", dpi=200, max_pages=5)
    assert img.size == (300, 400) and img.mode == "RGB"


@pytest.mark.parametrize(
    ("data", "ctype", "message"),
    [
        (_pdf(3), "application/pdf", "limit is 2"),
        (b"%PDF-1.4 not really a pdf", "application/pdf", "unreadable PDF"),
        (b"\x89PNG\r\n\x1a\n broken", "image/png", "unreadable image"),
        (b"GIF89a", "image/gif", "unsupported"),
    ],
)
def test_rasterise_rejects_bad_input_permanently(data: bytes, ctype: str, message: str) -> None:
    with pytest.raises(PermanentError, match=message):
        rasterise(data, ctype, dpi=150, max_pages=2)


# --- engine output parsing ----------------------------------------------------------------


def test_parse_tesseract_skips_layout_rows_and_groups_lines() -> None:
    data = {
        "text": ["", "Invoice", "No.", "", "Total", "1,180.00", " "],
        "conf": [-1, 96.0, 91.0, -1, 88.0, 72.5, 40.0],
        "block_num": [1, 1, 1, 2, 2, 2, 2],
        "par_num": [1, 1, 1, 1, 1, 1, 1],
        "line_num": [1, 1, 1, 1, 1, 1, 1],
        "left": [0, 10, 80, 0, 10, 90, 200],
        "top": [0, 20, 20, 0, 60, 60, 60],
        "width": [0, 60, 30, 0, 50, 70, 5],
        "height": [0, 15, 15, 0, 15, 15, 15],
    }
    result = parse_tesseract(data, 1000, 1400)
    assert [w.text for w in result.words] == ["Invoice", "No.", "Total", "1,180.00"]
    assert result.words[3].bbox == (90, 60, 160, 75)
    assert result.words[3].conf == pytest.approx(0.725)
    assert result.lines() == ["Invoice No.", "Total 1,180.00"]


def test_parse_textract_converts_relative_boxes_to_pixels() -> None:
    def word(i: str, text: str, left: float, conf: float) -> dict[str, object]:
        box = {"Left": left, "Top": 0.1, "Width": 0.1, "Height": 0.02}
        return {"Id": i, "BlockType": "WORD", "Text": text, "Confidence": conf,
                "Geometry": {"BoundingBox": box}}  # fmt: skip

    blocks = [
        {"Id": "p", "BlockType": "PAGE"},
        {
            "Id": "l1",
            "BlockType": "LINE",
            "Relationships": [{"Type": "CHILD", "Ids": ["w1", "w2"]}],
        },
        word("w1", "Total", 0.5, 99.0),
        word("w2", "1180.00", 0.62, 95.0),
    ]
    result = parse_textract(blocks, width=1000, height=2000)
    assert result.lines() == ["Total 1180.00"]
    assert result.words[0].bbox == (500, 200, 600, 240)
    assert result.words[1].conf == pytest.approx(0.95)


def test_ocr_result_json_round_trip() -> None:
    result = OcrResult(
        words=[OcrWord("Total", (1, 2, 3, 4), 0.9, line=0), OcrWord("10", (5, 2, 9, 4), 0.7, 0)],
        page_width=100,
        page_height=200,
        engine="tesseract",
    )
    again = OcrResult.from_json(result.to_json())
    assert again == result
    assert again.mean_conf == pytest.approx(0.8)
    assert OcrResult([], 1, 1, "x").mean_conf is None


def test_engine_factory() -> None:
    assert get_engine(Settings(ocr_engine="tesseract")).name == "tesseract"


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract binary not installed")
def test_tesseract_reads_real_text() -> None:
    img = Image.new("L", (900, 200), 255)
    ImageDraw.Draw(img).text((20, 60), "TOTAL 1180.00", fill=0, font_size=60)
    result = get_engine(Settings(ocr_engine="tesseract")).run(to_png(img))
    assert "1180.00" in " ".join(w.text for w in result.words)
