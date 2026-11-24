# Phase 2: Ingestion, queues and OCR

Phase 2 turns an upload into page images and OCR text with word boxes, through a queue-driven
pipeline in which every stage can fail, retry or receive the same message twice without
doing any harm.

```
POST /v1/documents ──▶ S3 raw/ + documents row ──▶ [ocr queue] ──▶ OCR worker
                                                                     │ page PNG + OCR JSON to S3
                                                                     │ pages rows, status ocr_done
                                                                     ▼
                                                               [extract queue]  (phase 3)
```

## 1. Upload API

`POST /v1/documents` (header `X-API-Key`):

- **Authentication.** The key is hashed and looked up through `auth_api_key()`, the narrow
  security-definer function from phase 1, because the tenant is not known yet.
- **Validation.** The file's type comes from its magic bytes (`%PDF-`, PNG and JPEG
  signatures), never from its name or the client's `Content-Type`. Size is capped by reading
  at most `MAX_UPLOAD_BYTES + 1` bytes.
- **Idempotent uploads.** The SHA-256 of the bytes is unique per tenant. A re-upload returns
  the existing document with HTTP 200 and `duplicate: true` and costs nothing. A race
  between two identical uploads is caught by the unique constraint and resolves the same way.
- **Ordering.** Store the original in S3, insert the row, enqueue the OCR message, then
  commit. If the commit fails after the enqueue, the worker finds no document and drops the
  message. The reverse order could strand a document in `uploaded` forever. This is a
  lightweight alternative to a transactional outbox, and it is safe because workers are
  idempotent.
- **404, not 403**, for another tenant's document: row-level security hides it, and the API
  never confirms that an id exists elsewhere.

## 2. One queue per stage

`ocr`, `extract`, `validate` and `index` each have their own queue and their own dead-letter
queue (`maxReceiveCount = 3`), so slow OCR never blocks extraction and each stage scales on
its own queue depth. Messages are tiny (`tenant_id`, `document_id`, `stage`); the data lives
in S3 and Postgres.

## 3. The worker contract (`apps/worker/src/docflow_worker/runner.py`)

1. **Claim** (short transaction): if `stage_runs` says this stage is `done`, it is a
   redelivery, so acknowledge and stop. Otherwise mark it `running` and count the attempt.
2. **Run** with no transaction open, so no locks are held during slow OCR. Outputs go to S3
   under deterministic keys, so a rerun overwrites them with the same content.
3. **Save** (second transaction): write rows, mark the stage `done`, advance the document
   status, enqueue the next stage, commit.
4. **Acknowledge** (delete the message) only after that commit.

| Failure | Behaviour |
|---|---|
| Transient (engine crash, timeout) | Retry with exponential backoff (10 s, 20 s, 40 s) by shortening the message's visibility timeout |
| Still failing on the 3rd receive | Document `failed` with the reason; the queue moves the message to the DLQ |
| Permanent (corrupt file, too many pages) | Document `failed` at once, no retries wasted |
| Poison message (unparseable body) | Never acknowledged, so it reaches the DLQ without touching any document |
| Message for a missing document | Dropped (an upload whose transaction rolled back) |
| Same message delivered twice | Second delivery sees `done` and acknowledges with no work |

All six paths are tested in `apps/worker/tests/test_runner.py` with an in-memory queue that
reproduces SQS's visibility, receive-count and redrive rules.

## 4. OCR

**Rasterise.** PDFs are rendered with PDFium (`pypdfium2`, a self-contained wheel) at
250 DPI, and images are decoded with EXIF rotation applied. Corrupt files, too many pages and
decompression bombs are permanent errors.

**Preprocess**, as pure functions:

- grayscale;
- median-filter denoise;
- deskew. A projection profile finds the angle at which rows of ink are sharpest, coarse
  then fine, using NumPy only.

**Erase ruling lines before OCR.** The first live run showed Tesseract dropping a whole
column of amounts inside a bordered totals box. The fix erases *thin* long lines (box
borders, table rules) but keeps *thick* solid areas, such as a black table-header bar and
its white text. Pixels are only whitened, never moved. So OCR reads the line-free image
while the stored page keeps its lines, and every box is valid on both.

Small bake-off on the first 9 synthetic invoices (78 known values: invoice number, date,
line amounts, subtotal, tax, total), Tesseract 5.5:

| Variant | Values found | Seconds per page |
|---|---|---|
| psm 3 (automatic layout) | 73/78, 93.6% | 0.46 |
| **psm 3 + line removal (chosen)** | **76/78, 97.4%** | 0.76 |
| psm 11 (sparse text) | 70/78, 89.7% | 0.47 |
| psm 6 + line removal | 76/78, 97.4% | 0.70 |

Nine documents is a sanity check, not a benchmark, and it proved too small: phase 3 measured
41 documents and switched to psm 6 with visual lines, which keeps table rows together (see
[phase 3](phase-3-extraction.md)). Phase 5 repeats this on the frozen golden
set, with Textract as the comparison.

**Engines** sit behind one `OcrEngine` interface: Tesseract (local, free) and AWS Textract
(paid comparison). `pages.ocr_engine` records which one read each page.

## 5. Synthetic invoices (`scripts/generate_invoices.py`)

Faker content and ReportLab layouts, with exact labels validated against `InvoiceV2`:

- Three layouts: classic tax invoice, modern invoice with a shaded header and a totals box,
  narrow receipt. Each uses different labels, date formats and number formats, including
  Indian lakh grouping (`1,03,095.00`).
- Valid GSTINs: the check character is computed with the same function the phase 4
  validator will use.
- Noise levels `clean` (PDF), `light` and `heavy` (JPEG with rotation, blur, speckle and
  compression).
- Deterministic: the same seed produces identical bytes (`invariant=1`).

## 6. Local AWS emulators

The design named LocalStack. Its current image exits with "License activation failed"
unless an account token is set, so the stack uses **S3Mock 5.2.3** (S3) and **ElasticMQ 1.7.1**
(SQS with redrive policies). Both are open source and pinned to exact versions. The code is
the same `boto3` code that runs on AWS; only the endpoint variables (`AWS_ENDPOINT_URL_S3`,
`AWS_ENDPOINT_URL_SQS`) differ. Presigned URLs are signed for `S3_PUBLIC_ENDPOINT_URL`
because the browser and the API container reach S3 at different hostnames.

## Deviations from the design doc

- `OcrWord` gains a `line` index: both engines report lines, and the prompt builder and
  chunker need them.
- Messages carry `tenant_id`, because the worker needs it to open a tenant-scoped session.
- Permanent errors skip retries; the design retried everything.
- The PaddleOCR adapter moves to the OCR bake-off (phase 7) so phase 2 ships only engines
  that are tested.
