# Phase 3: Extraction and confidence

Phase 3 turns OCR text into the typed invoice schema with an LLM. Every value gets a page, a
box and a confidence that has been checked against reality on a labelled dev set.

```
OCR JSON ─▶ prompt (template + line-tagged OCR) ─▶ gpt-oss-120b (strict JSON schema)
                                                       │ invalid? one retry with the error
                                                       ▼
            gpt-oss-20b, same prompt ─▶ agreement ─▶ grounding to OCR words ─▶ raw score
                                                                                  │
                                       isotonic calibration (fitted on dev set) ◀─┘
                                                       ▼
                         extraction row + usage_events ─▶ [validate queue]  (phase 4)
```

## 1. The schema is the contract

The Pydantic model `InvoiceV2` generates the JSON schema sent to the model and parses the
reply. `llm_schema.py` builds the **strict** form that structured output requires: every
property required, `additionalProperties: false`, no `$defs`. Amounts are strings
(`"1180.00"`) and are parsed straight into `Decimal`, so floats never touch money.

**Every field may be null, required ones included.** The first live run showed why. On an
invoice whose total was missing from the OCR text, the model correctly answered `null`.
The provider then rejected the reply, because the schema said `total` must be a string. Forcing a value
forces a guess, so "not printed" must always be a valid answer. "Required" is enforced
downstream: a required field that comes back null gets a confidence of 0.05 and goes to a
reviewer (and in phase 4, a failed `required_fields` rule). Replies are parsed with a
*lenient* copy of the schema, where all fields are optional but every type is still checked.

## 2. Prompt and versioning

The prompt has four parts in a fixed order:

1. system prompt (rules and schema);
2. few-shot examples (none yet; the phase 5 learning loop fills them);
3. the OCR text, each line tagged `[p1 l09]`;
4. the instruction to answer with JSON only.

`prompt_hash` is the hash of the **template**, which is everything except the document. Hashing the
whole prompt would make every document its own "version".

An **extractor version** is (doc type, schema version, model, prompt hash, few-shot policy,
config including calibration). The active row in `extractor_versions` is what the workers
run: settings only bootstrap version 1. Changing anything means registering a candidate and
promoting it (`scripts/admin.py`). Promotion retires the old version in the same
transaction, and nothing is ever deleted, so a rollback is just another promotion.
The worker logs `prompt_drift` if the code's template no longer matches the active version's
hash. That check fired during this phase, after the nullable-schema change, and version 2
was registered to fix it.

## 3. Provider client

One client for any OpenAI-compatible endpoint (Groq, Gemini's compatibility endpoint,
Ollama, paid providers). It records tokens, latency and cost for every call into
`usage_events`. The OpenAI SDK retries 429s and 5xx errors with backoff that honours
`retry-after`.

The model is chosen by measurement, not by name. The configured `llama-3.3-70b-versatile`
was no longer offered, so three available models were probed on the same strict-schema task:

| Model | Free limit | Strict JSON | Result |
|---|---|---|---|
| openai/gpt-oss-120b | 1,000 req/day, 8,000 tokens/min | clean | **extractor** |
| openai/gpt-oss-20b | same | clean | **second opinion** |
| qwen/qwen3.8-27b | same | malformed value (`"}47523.00"`) | rejected |

Average per document: about 1,240 tokens in and 440 to 510 out per call, 6.2 s for the 120b
model and 1.3 s for the 20b. Notional cost is about $0.0007 per page at list prices.

**Failure handling.** An invalid reply (bad JSON, a schema violation, or an HTTP 400 from the provider's
strict mode) gets exactly one retry, with the error appended. A second failure stores
`failed_parse` and sends the whole document to `needs_review`. Rate limits and network
errors are not parse failures: they are retried by the queue with backoff.

## 4. Grounding

Each value is matched back to OCR words on one line: amounts by `Decimal`, dates by parsed
date, currency by symbol or code, text by exact or fuzzy match (similarity of 0.8 or more). The match
gives the review UI a box, and gives confidence two signals. A value found nowhere is the
signature of a hallucination: it keeps no box and its confidence drops.

When a number is printed twice, labels break the tie. `total` prefers a line with "Total" but
not "Sub-total", and table cells prefer the line where their row's description was found.
Two guards came from tests: a bare `"10"` is not accepted as a date (dateutil would read it
as the 10th of this month), and shorter spans win ties.

## 5. Confidence

| Signal | Source |
|---|---|
| grounding | exact, fuzzy or not found on the page |
| OCR confidence | mean confidence of the matched words |
| agreement | did gpt-oss-20b extract the same value (the providers return no log-probabilities) |
| required missing | a required field came back null |
| validators | phase 4 |

`raw_score` combines them into a number that **ranks** fields. A rank is not a probability,
so an isotonic regression (pool-adjacent-violators, about 30 lines, no scikit-learn) fitted on
the dev set maps it to one. The review threshold is then the lowest score whose auto-approved
fields still reach the target precision.

Calibration needs no model calls. Each stored field keeps its `signals` and `raw_score`, so
`scripts/calibrate.py` refits from the database and labels alone.

## 6. Results on the dev set

Dev set: 41 synthetic invoices from three seeds (three layouts, clean, light and heavy noise),
1,001 fields. These documents tune OCR settings, calibration and the threshold, so they are
**not** the evaluation set. Phase 5 scores everything on a separate, frozen golden set.

### First run, and what the errors showed

Field accuracy was 93.6%, with 15 of 41 invoices entirely correct. Grouping the wrong fields
by cause found three problems, and the model caused none of them:

1. **OCR read tables by column.** Tesseract's default layout mode (psm 3) splits a page into
   column blocks, so Description, Qty and Amount came out as separate lists. On receipts it
   even split `20,312.88` into `20,312.` and `88`. The model was given scrambled tables.
2. **OCR swaps look-alike characters in tax IDs** (`O`/`0`, `Z`/`2`, `I`/`1`). Both models
   copy the misread faithfully, the value grounds exactly, so these are wrong fields with
   *high* confidence, the worst kind.
3. A bordered totals box hid the amounts (already fixed in phase 2 by erasing ruling lines).

### The OCR fix, measured on all 41 documents

| Tesseract setting | Amounts read whole | Table rows kept on one line | s/page |
|---|---|---|---|
| psm 3 (phase 2 choice) | 95.2% | 79.7% | 0.30 |
| psm 3 + visual lines | 95.2% | 98.1% | 0.30 |
| **psm 6 + visual lines (chosen)** | **98.9%** | **98.7%** | 0.28 |

"Visual lines" (`ocr/layout.py`) regroups words by vertical overlap into the rows a person
sees, whatever the engine did. It changes nothing for psm 6, which already reads in rows, but
it keeps the fix engine-agnostic: Textract also numbers lines per block. The phase 2 bake-off
on 9 documents had shown psm 6 and psm 3 tied, so the larger sample corrected that decision.

### After the fix (extractor version 2, same model and prompt)

| | Before | After |
|---|---|---|
| Field accuracy | 93.6% | **97.9%** |
| Invoices with every field right | 15/41 | **24/41** |
| total / subtotal / tax | 90.2% / 90.2% / 80.5% | **100% / 100% / 100%** |
| vendor_tax_id | 63.4% | 65.9% |

### Calibration and threshold

Fields scored 0.95 or more by the raw signals were right 99.5% of the time (929 fields).
Fields below that were right much less often, so the signals do separate good from bad. The
review threshold depends on how many wrong fields the business accepts skipping review:

| Target precision | Threshold | Auto-approved | Wrong fields let through |
|---|---|---|---|
| 98.0% | 0.5000 | 99.6% | 17 |
| 99.0% | 0.8824 | 97.1% | 8 |
| **99.5% (chosen)** | **0.9286** | **91.2%** | **3** |
| 99.9% | none | 0% | 0 |

At 99.5%, nine fields in ten skip review. 99.9% is not reachable yet: with 41 documents the
calibration cannot certify that much confidence, the coarse-threshold limitation the design
doc predicts for small dev sets. The calibration and threshold are stored on **extractor
version 3** (`eval/calibration/invoice-64b8483e.json`), which was promoted over version 2.

### What is left for phase 4

The remaining big error is the tax ID: 14 of 41 wrong, at high confidence, because every
current signal agrees with the misread. A GSTIN check character catches each one. That is
the `tax_id_format` validator, and its result feeds `validators_passed` into this confidence
score, which is why the signal slot already exists.

## Deviations from the design doc

- `prompt_hash` hashes the template, not the per-document prompt.
- Model signal is cross-model agreement: the providers tried return no log-probabilities.
- All fields are nullable at the model boundary; "required" is enforced as confidence.
- Calibration is stored on the extractor version (`config.calibration`), so changing it is a
  versioned, reversible promotion like any other change.
